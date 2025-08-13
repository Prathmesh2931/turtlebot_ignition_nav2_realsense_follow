#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
import struct
from std_msgs.msg import Float32
import math
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

class BoxDistanceTracker(Node):
    def __init__(self):
        super().__init__('box_distance_tracker')
        
        # Create CV bridge
        self.bridge = CvBridge()
        
        # Create subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.rgb_callback,
            10)
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10)
        
        # Create publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        
        # Create publisher for distance (useful for debugging)
        self.distance_pub = self.create_publisher(
            Float32,
            '/box_distance',
            10)
        
        # Initialize variables
        self.rgb_image = None
        self.depth_image = None
        self.target_distance = 0.02  # 2 cm target distance
        self.box_detected = False
        self.box_distance = float('inf')
        self.box_position = [0, 0]  # [x, y] in image coordinates
        self.box_dimensions = [0.1, 0.1, 0.1]  # 10cm box dimensions
        
        # Box color range in HSV (red box)
        # Adjust these values based on your actual box color
        self.lower_color = np.array([0, 100, 100])   # Lower red
        self.upper_color = np.array([10, 255, 255])  # Upper red
        
        self.lower_color2 = np.array([160, 100, 100])  # Lower red (second range)
        self.upper_color2 = np.array([180, 255, 255])  # Upper red (second range)
        
        # Control loop timer (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Box distance tracker initialized')
    
    def rgb_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # Detect box in RGB image
            self.detect_box()
        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {str(e)}')
    
    def depth_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')
    
    def detect_box(self):
        if self.rgb_image is None:
            return
        
        try:
            # Convert BGR to HSV
            hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
            
            # Threshold the HSV image to get only red colors
            # Red color has two ranges in HSV, so we need two masks
            mask1 = cv2.inRange(hsv, self.lower_color, self.upper_color)
            mask2 = cv2.inRange(hsv, self.lower_color2, self.upper_color2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (assumed to be the box)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # Only process if the contour is large enough
                if area > 500:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Calculate center of the box
                    box_center_x = x + w // 2
                    box_center_y = y + h // 2
                    
                    # Update box position
                    self.box_position = [box_center_x, box_center_y]
                    self.box_detected = True
                    
                    # Get distance from depth image if available
                    if self.depth_image is not None:
                        # Get dimensions of depth image
                        depth_h, depth_w = self.depth_image.shape
                        
                        # Scale RGB image coordinates to depth image coordinates
                        # This is needed if RGB and depth images have different resolutions
                        rgb_h, rgb_w = self.rgb_image.shape[:2]
                        depth_x = int(box_center_x * depth_w / rgb_w)
                        depth_y = int(box_center_y * depth_h / rgb_h)
                        
                        # Ensure coordinates are within bounds
                        depth_x = min(max(depth_x, 0), depth_w - 1)
                        depth_y = min(max(depth_y, 0), depth_h - 1)
                        
                        # Get distance at box center
                        # Take average of small region around center for stability
                        region_size = 5
                        x_start = max(0, depth_x - region_size)
                        x_end = min(depth_w, depth_x + region_size)
                        y_start = max(0, depth_y - region_size)
                        y_end = min(depth_h, depth_y + region_size)
                        
                        depth_region = self.depth_image[y_start:y_end, x_start:x_end]
                        valid_depths = depth_region[~np.isnan(depth_region) & ~np.isinf(depth_region) & (depth_region > 0)]
                        
                        if valid_depths.size > 0:
                            # Calculate median depth for robustness
                            distance = np.median(valid_depths)
                            self.box_distance = float(distance)
                            
                            # Publish distance for debugging
                            distance_msg = Float32()
                            distance_msg.data = self.box_distance
                            self.distance_pub.publish(distance_msg)
                            
                            self.get_logger().info(f'Box detected at distance: {self.box_distance:.3f}m')
                        else:
                            self.get_logger().warn('No valid depth measurements for the box')
                    
                    # Optional: Draw box on image for visualization (if you want to debug)
                    debug_image = self.rgb_image.copy()
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(debug_image, f'Distance: {self.box_distance:.3f}m', 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # If you want to visualize for debugging, you could publish this image
                    # through another publisher
                else:
                    # No box detected that meets the area threshold
                    self.box_detected = False
            else:
                # No contours found
                self.box_detected = False
        
        except Exception as e:
            self.get_logger().error(f'Error in box detection: {str(e)}')
    
    def control_loop(self):
        # Create velocity command
        vel_msg = Twist()
        
        # If box is detected
        if self.box_detected:
            # Calculate distance to maintain (box radius + target distance)
            # Box radius is approximately half the box width (10cm/2 = 5cm)
            box_radius = 0.05  # 5 cm radius for a 10cm box
            desired_distance = box_radius + self.target_distance  # 5cm + 2cm = 7cm
            
            # If we're further than the desired distance, move toward the box
            if self.box_distance > desired_distance + 0.01:  # 1cm tolerance
                # Calculate speed based on distance (slow down as we get closer)
                speed = min(0.2, max(0.05, (self.box_distance - desired_distance) * 0.5))
                
                # Set linear velocity to approach the box
                vel_msg.linear.x = speed
                
                # Calculate angular velocity to center the box in the image
                if self.rgb_image is not None:
                    image_width = self.rgb_image.shape[1]
                    image_center_x = image_width // 2
                    
                    # Calculate error in pixels
                    error_x = self.box_position[0] - image_center_x
                    
                    # Convert to angular velocity (scale factor determined empirically)
                    angular_velocity = -error_x * 0.001
                    vel_msg.angular.z = angular_velocity
                
                self.get_logger().info(f'Moving toward box. Distance: {self.box_distance:.3f}m, Target: {desired_distance:.3f}m, Speed: {speed:.3f}m/s')
            elif self.box_distance < desired_distance - 0.01:  # Too close, back up
                # Calculate speed based on distance
                speed = min(0.1, max(0.05, (desired_distance - self.box_distance) * 0.5))
                
                # Set linear velocity to back away from the box
                vel_msg.linear.x = -speed
                
                self.get_logger().info(f'Backing away from box. Distance: {self.box_distance:.3f}m, Target: {desired_distance:.3f}m, Speed: {-speed:.3f}m/s')
            else:
                # We're at the desired distance, stop
                self.get_logger().info(f'At target distance: {self.box_distance:.3f}m')
                
                # Refine position to center on the box
                if self.rgb_image is not None:
                    image_width = self.rgb_image.shape[1]
                    image_center_x = image_width // 2
                    
                    # Calculate error in pixels
                    error_x = self.box_position[0] - image_center_x
                    
                    # If the box is not centered, adjust orientation slightly
                    if abs(error_x) > 20:  # 20-pixel threshold
                        # Convert to angular velocity (reduced factor for fine adjustment)
                        angular_velocity = -error_x * 0.0005
                        vel_msg.angular.z = angular_velocity
        else:
            # If no box is detected, perform a slow search rotation
            vel_msg.angular.z = 0.2
            self.get_logger().info('No box detected. Searching...')
        
        # Publish velocity command
        self.cmd_vel_pub.publish(vel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BoxDistanceTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure to stop the robot when the node shuts down
        stop_msg = Twist()
        node.cmd_vel_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()