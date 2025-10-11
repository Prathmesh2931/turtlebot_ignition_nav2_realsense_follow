#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool, Float32
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class EnhancedYOLOTracker(Node):
    def __init__(self):
        super().__init__('enhanced_yolo_tracker')
        
        # QoS Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Initialize YOLO model
        self.yolo_model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        
        # Subscribers
        self.rgb_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.rgb_callback, qos_profile)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos_profile)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        
        # Voice command subscribers
        self.target_object_sub = self.create_subscription(String, '/target_object', self.target_object_callback, 10)
        self.control_mode_sub = self.create_subscription(String, '/control_mode', self.control_mode_callback, 10)
        self.search_mode_sub = self.create_subscription(Bool, '/search_mode', self.search_mode_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.detection_pub = self.create_publisher(Bool, '/object_detected', 10)
        self.distance_pub = self.create_publisher(Float32, '/object_distance', 10)
        
        # State variables
        self.rgb_image = None
        self.depth_image = None
        self.laser_data = None
        
        # Tracking variables
        self.target_object = "person"
        self.tracking_mode = "yolo"  # "yolo" or "color"
        self.control_mode = "searching"  # "searching", "following", "stopped", "paused"
        self.search_mode = True
        
        # Detection variables
        self.object_detected = False
        self.object_distance = float('inf')
        self.object_position = [0, 0]
        self.detection_confidence = 0.0
        self.min_confidence = 0.5
        self.target_distance = 1.5
        
        # Movement parameters
        self.max_linear_speed = 0.3
        self.max_angular_speed = 0.5
        self.search_angular_speed = 0.3
        
        # Safety
        self.min_obstacle_distance = 0.5
        self.emergency_stop = False
        
        # Color tracking parameters (for red/blue/green objects)
        self.color_ranges = {
            "red_object": {
                "lower1": np.array([0, 100, 100]),
                "upper1": np.array([10, 255, 255]),
                "lower2": np.array([160, 100, 100]),
                "upper2": np.array([180, 255, 255])
            },
            "blue_object": {
                "lower1": np.array([100, 100, 100]),
                "upper1": np.array([130, 255, 255]),
                "lower2": None,
                "upper2": None
            },
            "green_object": {
                "lower1": np.array([40, 100, 100]),
                "upper1": np.array([80, 255, 255]),
                "lower2": None,
                "upper2": None
            }
        }
        
        # Control timer
        self.timer = self.create_timer(0.05, self.control_loop)
        
        # YOLO class names
        self.class_names = self.yolo_model.names
        
        self.get_logger().info(f'Enhanced YOLO Tracker initialized')
        self.get_logger().info(f'Available YOLO classes: {list(self.class_names.values())}')
        
    def target_object_callback(self, msg):
        """Update target object from voice command"""
        new_target = msg.data.lower()
        
        # Check if it's a color-based object
        if new_target in self.color_ranges:
            self.target_object = new_target
            self.tracking_mode = "color"
            self.get_logger().info(f'Switching to color tracking: {new_target}')
        else:
            # Check if it's a valid YOLO class
            for class_id, class_name in self.class_names.items():
                if new_target in class_name.lower() or class_name.lower() in new_target:
                    self.target_object = class_name
                    self.tracking_mode = "yolo"
                    self.get_logger().info(f'Switching to YOLO tracking: {class_name}')
                    return
            
            self.get_logger().warn(f'Unknown object: {new_target}')
    
    def control_mode_callback(self, msg):
        """Update control mode"""
        self.control_mode = msg.data
        self.get_logger().info(f'Control mode: {self.control_mode}')
    
    def search_mode_callback(self, msg):
        """Update search mode"""
        self.search_mode = msg.data
        if self.search_mode:
            self.control_mode = "searching"
    
    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.detect_objects()
        except Exception as e:
            self.get_logger().error(f'RGB callback error: {e}')
    
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f'Depth callback error: {e}')
    
    def scan_callback(self, msg):
        self.laser_data = msg
        self.check_obstacles()
    
    def detect_objects(self):
        """Detect objects using YOLO or color detection"""
        if self.rgb_image is None:
            return
        
        if self.tracking_mode == "yolo":
            self.detect_yolo_objects()
        elif self.tracking_mode == "color":
            self.detect_color_objects()
    
    def detect_yolo_objects(self):
        """YOLO-based object detection"""
        try:
            results = self.yolo_model(self.rgb_image)
            
            self.object_detected = False
            best_detection = None
            best_confidence = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.class_names[class_id]
                        
                        if (class_name.lower() == self.target_object.lower() and 
                            confidence > self.min_confidence and
                            confidence > best_confidence):
                            
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            best_detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                            }
                            best_confidence = confidence
            
            if best_detection:
                self.object_detected = True
                self.object_position = best_detection['center']
                self.detection_confidence = best_confidence
                self.calculate_object_distance()
                self.draw_yolo_detection(best_detection)
                
        except Exception as e:
            self.get_logger().error(f'YOLO detection error: {e}')
    
    def detect_color_objects(self):
        """Color-based object detection"""
        try:
            if self.target_object not in self.color_ranges:
                return
            
            hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
            color_range = self.color_ranges[self.target_object]
            
            # Create mask
            mask1 = cv2.inRange(hsv, color_range["lower1"], color_range["upper1"])
            if color_range["lower2"] is not None:
                mask2 = cv2.inRange(hsv, color_range["lower2"], color_range["upper2"])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = mask1
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            self.object_detected = False
            if contours:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 1000:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    self.object_detected = True
                    self.object_position = [center_x, center_y]
                    self.detection_confidence = min(area / 10000.0, 1.0)  # Normalize area to confidence
                    self.calculate_object_distance()
                    
                    # Draw detection
                    cv2.rectangle(self.rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(self.rgb_image, f'{self.target_object}: {self.detection_confidence:.2f}', 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            self.get_logger().error(f'Color detection error: {e}')
    
    def calculate_object_distance(self):
        """Calculate distance to detected object"""
        if self.depth_image is None or not self.object_detected:
            return
        
        try:
            center_x = int(self.object_position[0])
            center_y = int(self.object_position[1])
            
            # Sample depth values around center
            sample_size = 10
            depth_values = []
            
            for dx in range(-sample_size, sample_size + 1):
                for dy in range(-sample_size, sample_size + 1):
                    x = center_x + dx
                    y = center_y + dy
                    
                    if (0 <= x < self.depth_image.shape[1] and 
                        0 <= y < self.depth_image.shape[0]):
                        depth_val = self.depth_image[y, x]
                        if not np.isnan(depth_val) and depth_val > 0:
                            depth_values.append(depth_val)
            
            if depth_values:
                self.object_distance = np.median(depth_values)
                
                # Publish distance
                distance_msg = Float32()
                distance_msg.data = self.object_distance
                self.distance_pub.publish(distance_msg)
                
        except Exception as e:
            self.get_logger().error(f'Distance calculation error: {e}')
    
    def draw_yolo_detection(self, detection):
        """Draw YOLO detection on image"""
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(self.rgb_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        label = f'{self.target_object}: {detection["confidence"]:.2f}'
        if self.object_distance < float('inf'):
            label += f' | {self.object_distance:.2f}m'
        
        cv2.putText(self.rgb_image, label, (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def check_obstacles(self):
        """Check for obstacles using laser scan"""
        if self.laser_data is None:
            return
        
        ranges = np.array(self.laser_data.ranges)
        front_ranges = np.concatenate([ranges[-30:], ranges[:30]])
        
        valid_ranges = front_ranges[np.isfinite(front_ranges)]
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.emergency_stop = min_distance < self.min_obstacle_distance
    
    def control_loop(self):
        """Main control loop"""
        if self.emergency_stop or self.control_mode == "stopped":
            self.stop_robot()
            return
        
        if self.control_mode == "paused":
            self.stop_robot()
            return
        
        # Publish detection status
        detection_msg = Bool()
        detection_msg.data = self.object_detected
        self.detection_pub.publish(detection_msg)
        
        if self.control_mode == "searching" or (self.control_mode == "following" and not self.object_detected):
            self.search_behavior()
        elif self.control_mode == "following" and self.object_detected:
            self.follow_behavior()
    
    def search_behavior(self):
        """Search for target object by rotating"""
        twist = Twist()
        twist.angular.z = self.search_angular_speed
        self.cmd_vel_pub.publish(twist)
    
    def follow_behavior(self):
        """Follow detected object"""
        if not self.object_detected:
            return
        
        twist = Twist()
        
        # Get image center
        image_center_x = self.rgb_image.shape[1] // 2
        
        # Calculate angular error
        angular_error = (self.object_position[0] - image_center_x) / image_center_x
        
        # Control angular velocity
        twist.angular.z = -angular_error * self.max_angular_speed
        
        # Control linear velocity based on distance
        if abs(angular_error) < 0.2:  # Only move when aligned
            distance_error = self.object_distance - self.target_distance
            if distance_error > 0.3:
                twist.linear.x = min(distance_error * 0.5, self.max_linear_speed)
            elif distance_error < -0.3:
                twist.linear.x = max(distance_error * 0.5, -self.max_linear_speed * 0.5)
        
        # Limit velocities
        twist.linear.x = max(min(twist.linear.x, self.max_linear_speed), -self.max_linear_speed)
        twist.angular.z = max(min(twist.angular.z, self.max_angular_speed), -self.max_angular_speed)
        
        self.cmd_vel_pub.publish(twist)
    
    def stop_robot(self):
        """Stop the robot"""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = EnhancedYOLOTracker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()