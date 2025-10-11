#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
import struct

class CompressedSensingFusion(Node):
    def __init__(self):
        super().__init__('cs_sensor_fusion')
        
        # CS Parameters - Reduced for memory efficiency
        self.compression_ratio = 0.1  # Use only 10% of original data to avoid memory issues
        self.patch_size = 16  # Larger patches for efficiency
        self.max_depth_points = 10000  # Limit depth points processed
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.rgb_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Publishers
        self.fused_features_pub = self.create_publisher(PointCloud2, '/fused_features', 10)
        self.compressed_rgb_pub = self.create_publisher(Image, '/compressed_rgb', 10)
        
        # Data buffers
        self.rgb_data = None
        self.depth_data = None
        self.lidar_data = None
        self.lidar_angles = None
        
        # Processed data
        self.rgb_compressed = None
        self.depth_compressed = None
        self.lidar_compressed = None
        
        # Camera intrinsics (adjust for your camera)
        self.fx = 554.0
        self.fy = 554.0
        self.cx = 320.0
        self.cy = 240.0
        
        # Timer for fusion - reduced frequency to avoid overwhelming
        self.timer = self.create_timer(0.2, self.fusion_callback)
        
        self.get_logger().info('Compressed Sensing Sensor Fusion initialized')
    
    def rgb_callback(self, msg):
        try:
            self.rgb_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.process_rgb_compressed()
        except Exception as e:
            self.get_logger().error(f'RGB processing error: {str(e)}')
    
    def depth_callback(self, msg):
        try:
            self.depth_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            self.process_depth_compressed()
        except Exception as e:
            self.get_logger().error(f'Depth processing error: {str(e)}')
    
    def scan_callback(self, msg):
        self.lidar_data = np.array(msg.ranges)
        self.lidar_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        self.process_lidar_compressed()
    
    def generate_sparse_measurement_matrix(self, signal_length, compressed_length):
        """Generate sparse measurement matrix for CS to save memory"""
        # Create sparse random matrix
        density = 0.1  # 10% non-zero elements
        M = sp.random(compressed_length, signal_length, density=density, format='csr')
        # Normalize rows
        row_norms = sp.linalg.norm(M, axis=1)
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        M = M.multiply(1.0 / row_norms[:, np.newaxis])
        return M
    
    def process_rgb_compressed(self):
        if self.rgb_data is None:
            return
        
        try:
            # Downsample for efficiency
            h, w = self.rgb_data.shape[:2]
            if h > 240 or w > 320:
                scale_factor = min(240/h, 320/w)
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                rgb_resized = cv2.resize(self.rgb_data, (new_w, new_h))
            else:
                rgb_resized = self.rgb_data.copy()
            
            # Simple compression using downsampling and quantization
            # This is more memory-efficient than patch-based CS
            compressed = cv2.resize(rgb_resized, None, 
                                  fx=self.compression_ratio, 
                                  fy=self.compression_ratio, 
                                  interpolation=cv2.INTER_AREA)
            
            # Quantize colors to reduce data
            compressed = (compressed // 32) * 32  # Reduce to 8 levels per channel
            
            # Upscale back for visualization
            self.rgb_compressed = cv2.resize(compressed, 
                                           (rgb_resized.shape[1], rgb_resized.shape[0]), 
                                           interpolation=cv2.INTER_LINEAR)
            
            # Publish compressed RGB
            compressed_msg = self.bridge.cv2_to_imgmsg(self.rgb_compressed.astype(np.uint8), encoding="bgr8")
            self.compressed_rgb_pub.publish(compressed_msg)
            
        except Exception as e:
            self.get_logger().warn(f'RGB compression failed: {str(e)}')
    
    def process_depth_compressed(self):
        if self.depth_data is None:
            return
        
        try:
            # Remove invalid depth values
            valid_depth = self.depth_data.copy()
            valid_depth[np.isnan(valid_depth) | np.isinf(valid_depth) | (valid_depth <= 0)] = 0
            
            # Downsample depth image to manageable size
            h, w = valid_depth.shape
            if h > 120 or w > 160:
                scale_factor = min(120/h, 160/w)
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                depth_resized = cv2.resize(valid_depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                depth_resized = valid_depth
            
            # Simple compression by keeping only every nth pixel
            step = int(1 / self.compression_ratio)
            compressed_indices = np.arange(0, depth_resized.size, step)
            
            # Store compressed depth as 1D array with spatial indices
            flat_depth = depth_resized.flatten()
            self.depth_compressed = flat_depth[compressed_indices]
            self.depth_indices = compressed_indices
            self.depth_shape = depth_resized.shape
            
        except Exception as e:
            self.get_logger().warn(f'Depth compression failed: {str(e)}')
    
    def process_lidar_compressed(self):
        if self.lidar_data is None:
            return
        
        try:
            # Remove invalid LIDAR readings
            valid_mask = np.isfinite(self.lidar_data) & (self.lidar_data > 0.1) & (self.lidar_data < 10.0)
            valid_ranges = self.lidar_data[valid_mask]
            valid_angles = self.lidar_angles[valid_mask] if self.lidar_angles is not None else np.arange(len(self.lidar_data))[valid_mask]
            
            if len(valid_ranges) > 0:
                # Compress by keeping every nth point
                step = max(1, int(1 / self.compression_ratio))
                compressed_indices = np.arange(0, len(valid_ranges), step)
                
                self.lidar_compressed = valid_ranges[compressed_indices]
                self.lidar_angles_compressed = valid_angles[compressed_indices]
            
        except Exception as e:
            self.get_logger().warn(f'LIDAR compression failed: {str(e)}')
    
    def fusion_callback(self):
        """Fuse compressed sensor data"""
        fused_points = []
        
        try:
            # Process RGB-D points
            if (hasattr(self, 'rgb_compressed') and self.rgb_compressed is not None and 
                hasattr(self, 'depth_compressed') and self.depth_compressed is not None):
                
                rgb_h, rgb_w = self.rgb_compressed.shape[:2]
                
                # Convert compressed depth back to spatial coordinates
                for i, depth_idx in enumerate(self.depth_indices[:min(len(self.depth_indices), 1000)]):  # Limit points
                    if i >= len(self.depth_compressed):
                        break
                        
                    depth_val = self.depth_compressed[i]
                    if depth_val > 0:
                        # Convert 1D index back to 2D coordinates
                        v = depth_idx // self.depth_shape[1]
                        u = depth_idx % self.depth_shape[1]
                        
                        # Scale coordinates to match RGB image
                        u_rgb = int(u * rgb_w / self.depth_shape[1])
                        v_rgb = int(v * rgb_h / self.depth_shape[0])
                        
                        if 0 <= u_rgb < rgb_w and 0 <= v_rgb < rgb_h:
                            # Convert to 3D coordinates
                            x = (u - self.depth_shape[1]/2) * depth_val / self.fx
                            y = (v - self.depth_shape[0]/2) * depth_val / self.fy
                            z = depth_val
                            
                            # Get RGB values
                            b, g, r = self.rgb_compressed[v_rgb, u_rgb]
                            
                            fused_points.append([x, y, z, float(r), float(g), float(b)])
            
            # Process LIDAR points
            if (hasattr(self, 'lidar_compressed') and self.lidar_compressed is not None and
                hasattr(self, 'lidar_angles_compressed') and self.lidar_angles_compressed is not None):
                
                for range_val, angle in zip(self.lidar_compressed[:500], self.lidar_angles_compressed[:500]):  # Limit points
                    x = range_val * np.cos(angle)
                    y = range_val * np.sin(angle)
                    z = 0.0
                    
                    # LIDAR points in red for distinction
                    fused_points.append([x, y, z, 255.0, 0.0, 0.0])
            
            # Publish fused point cloud
            if fused_points:
                self.publish_fused_pointcloud(fused_points)
                
        except Exception as e:
            self.get_logger().error(f'Fusion error: {str(e)}')
    
    def publish_fused_pointcloud(self, points):
        """Publish fused point cloud with correct header"""
        try:
            # Create proper header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "camera_link"
            
            # Create PointCloud2 message
            fields = [
                pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='r', offset=12, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='g', offset=16, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='b', offset=20, datatype=pc2.PointField.FLOAT32, count=1),
            ]
            
            # Convert points to proper format
            cloud_data = []
            for point in points:
                # Pack each point as binary data
                cloud_data.append(struct.pack('ffffff', *point))
            
            # Create point cloud message
            pc_msg = PointCloud2()
            pc_msg.header = header
            pc_msg.height = 1
            pc_msg.width = len(points)
            pc_msg.fields = fields
            pc_msg.is_bigendian = False
            pc_msg.point_step = 24  # 6 floats * 4 bytes each
            pc_msg.row_step = pc_msg.point_step * pc_msg.width
            pc_msg.data = b''.join(cloud_data)
            pc_msg.is_dense = True
            
            self.fused_features_pub.publish(pc_msg)
            
        except Exception as e:
            self.get_logger().error(f'PointCloud publish error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = CompressedSensingFusion()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()