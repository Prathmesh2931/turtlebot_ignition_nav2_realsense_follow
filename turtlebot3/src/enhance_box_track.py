#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
import struct
from std_msgs.msg import Float32
import math
import time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class EnhancedBoxTracker(Node):
    def __init__(self):
        super().__init__('enhanced_box_tracker')
        
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
        
        # Subscribe to laser scan for collision avoidance
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
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
        self.laser_data = None
        self.target_distance = 0.02  # 2 cm target distance
        self.box_detected = False
        self.box_distance = float('inf')
        self.box_position = [0, 0]  # [x, y] in image coordinates
        self.box_width = 0
        self.box_height = 0
        self.box_dimensions = [0.1, 0.1, 0.1]  # 10cm box dimensions
        
        # Search pattern variables
        self.search_state = "SCAN"  # SCAN, MOVE, APPROACH
        self.scan_start_time = None
        self.scan_duration = 10.0  # seconds for a full 360° scan
        self.move_distance = 0.5   # meters to move if no box found
        self.move_start_time = None
        self.move_direction = 0    # 0: forward, 1: right, 2: backward, 3: left
        self.search_pattern_count = 0
        
        # Alignment variables
        self.alignment_threshold = 5.0  # degrees
        self.is_aligned = False
        
        # Box color range in HSV (red box)
        # Adjust these values based on your actual box color
        self.lower_color = np.array([0, 100, 100])   # Lower red
        self.upper_color = np.array([10, 255, 255])  # Upper red
        
        self.lower_color2 = np.array([160, 100, 100])  # Lower red (second range)
        self.upper_color2 = np.array([180, 255, 255])  # Upper red (second range)
        
        # Control loop timer (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Enhanced box tracker initialized')
    
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
    
    def scan_callback(self, msg):
        self.laser_data = msg
    
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
                    self.box_width = w
                    self.box_height = h
                    
                    # Calculate center of the box
                    box_center_x = x + w // 2
                    box_center_y = y + h // 2
                    
                    # Update box position
                    self.box_position = [box_center_x, box_center_y]
                    self.box_detected = True
                    
                    # Calculate box orientation (to align 90 degrees with it)
                    # For a square box, we analyze if it appears as a proper square in the image
                    # If box is rotated, the aspect ratio might change
                    aspect_ratio = float(w) / h
                    
                    # Check if we're roughly aligned at 90 degrees (square should appear square-ish)
                    self.is_aligned = 0.85 < aspect_ratio < 1.15
                    
                    # Get distance from depth image if available
                    if self.depth_image is not None:
                        # Get dimensions of depth image
                        depth_h, depth_w = self.depth_image.shape
                        
                        # Scale RGB image coordinates to depth image coordinates
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
                            
                            # Change state to APPROACH when box is detected
                            if self.search_state in ["SCAN", "MOVE"]:
                                self.search_state = "APPROACH"
                            
                            self.get_logger().info(f'Box detected at distance: {self.box_distance:.3f}m, Aligned: {self.is_aligned}')
                        else:
                            self.get_logger().warn('No valid depth measurements for the box')
                    
                    # Optional: Draw box on image for visualization
                    debug_image = self.rgb_image.copy()
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(debug_image, f'Distance: {self.box_distance:.3f}m', 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # If you want to visualize for debugging, you could publish this image
                else:
                    # No box detected that meets the area threshold
                    self.box_detected = False
            else:
                # No contours found
                self.box_detected = False
        
        except Exception as e:
            self.get_logger().error(f'Error in box detection: {str(e)}')
    
    def is_path_clear(self):
        """Check if the path in front of the robot is clear"""
        if self.laser_data is None:
            return False
        
        # Check the forward-facing laser scan points
        # Assuming the laser provides 360 degrees scan data
        front_angles = np.where(
            (np.array(self.laser_data.ranges) > self.laser_data.angle_min + math.pi*7/8) | 
            (np.array(self.laser_data.ranges) < self.laser_data.angle_min + math.pi/8)
        )[0]
        
        front_distances = np.array(self.laser_data.ranges)[front_angles]
        min_distance = np.min(front_distances) if front_distances.size > 0 else float('inf')
        
        # Return True if the minimum distance is greater than the threshold
        return min_distance > 0.5  # 50cm threshold
    
    def get_safe_direction(self):
        """Find a safe direction to move in (returns angle in radians)"""
        if self.laser_data is None:
            return 0.0
        
        # Convert laser scan to numpy array
        ranges = np.array(self.laser_data.ranges)
        
        # Replace inf values with a large number
        ranges[np.isinf(ranges)] = 10.0
        
        # Find direction with maximum distance
        max_idx = np.argmax(ranges)
        angle = self.laser_data.angle_min + max_idx * self.laser_data.angle_increment
        
        return angle
    
    def execute_scan(self):
        """Rotate the robot 360 degrees to search for the box"""
        vel_msg = Twist()
        
        # Initialize scan if it's the first time
        if self.scan_start_time is None:
            self.scan_start_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.get_logger().info('Starting 360-degree scan')
        
        # Calculate elapsed time
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed = current_time - self.scan_start_time
        
        # If we haven't completed a full 360-degree scan
        if elapsed < self.scan_duration:
            # Rotate at a constant speed to complete 360 degrees in scan_duration seconds
            angular_velocity = 2 * math.pi / self.scan_duration  # radians/second
            vel_msg.angular.z = angular_velocity
        else:
            # Scan complete, move to next state
            self.get_logger().info('Scan complete, box not found. Moving to new position.')
            self.scan_start_time = None
            self.search_state = "MOVE"
            vel_msg.angular.z = 0.0
        
        return vel_msg
    
    def execute_move(self):
        """Move the robot to a new position when box is not found during scan"""
        vel_msg = Twist()
        
        # Initialize move if it's the first time
        if self.move_start_time is None:
            self.move_start_time = self.get_clock().now().seconds_nanoseconds()[0]
            
            # Choose a direction based on the search pattern
            # We use a spiral pattern: forward, right, backward (longer), left (longer), etc.
            self.move_direction = self.search_pattern_count % 4
            self.search_pattern_count += 1
            
            # Increase move distance every two moves
            if self.search_pattern_count % 2 == 0:
                self.move_distance += 0.2
            
            self.get_logger().info(f'Moving to new position, direction: {self.move_direction}, distance: {self.move_distance}m')
        
        # Check if path is clear in the desired direction
        if not self.is_path_clear():
            # Path not clear, find a safe direction
            safe_angle = self.get_safe_direction()
            vel_msg.angular.z = 0.2 if safe_angle > 0 else -0.2
            self.get_logger().info(f'Path not clear, turning to safe direction: {safe_angle}')
            return vel_msg
        
        # Calculate elapsed time
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed = current_time - self.move_start_time
        
        # Calculate how long we need to move to cover the desired distance
        # Assuming a linear velocity of 0.2 m/s
        linear_velocity = 0.2
        move_duration = self.move_distance / linear_velocity
        
        # If we haven't moved enough yet
        if elapsed < move_duration:
            # Set velocity based on direction
            if self.move_direction == 0:  # Forward
                vel_msg.linear.x = linear_velocity
            elif self.move_direction == 1:  # Right
                vel_msg.angular.z = -linear_velocity
            elif self.move_direction == 2:  # Backward
                vel_msg.linear.x = -linear_velocity
            elif self.move_direction == 3:  # Left
                vel_msg.angular.z = linear_velocity
        else:
            # Move complete, back to scanning
            self.get_logger().info('Move complete. Starting new scan.')
            self.move_start_time = None
            self.search_state = "SCAN"
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
        
        return vel_msg
    
    def execute_approach(self):
        """Approach the detected box and align with it"""
        vel_msg = Twist()
        
        # Calculate distance to maintain (box radius + target distance)
        # Box radius is approximately half the box width (10cm/2 = 5cm)
        box_radius = 0.05  # 5 cm radius for a 10cm box
        desired_distance = box_radius + self.target_distance  # 5cm + 2cm = 7cm
        
        # If we're not aligned with the box at 90 degrees
        if not self.is_aligned and self.box_width > 0 and self.box_height > 0:
            # If the box appears wider than tall, we need to rotate to align
            aspect_ratio = float(self.box_width) / self.box_height
            
            if aspect_ratio > 1.15:  # Box is wider than tall
                vel_msg.angular.z = 0.1  # Rotate clockwise
                self.get_logger().info('Aligning with box: rotating clockwise')
                return vel_msg
            elif aspect_ratio < 0.85:  # Box is taller than wide
                vel_msg.angular.z = -0.1  # Rotate counter-clockwise
                self.get_logger().info('Aligning with box: rotating counter-clockwise')
                return vel_msg
            else:
                # We're approximately aligned
                self.is_aligned = True
                self.get_logger().info('Box alignment achieved')
        
        # Now handle the distance
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
                
                # Convert to angular velocity
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
        
        # If box is lost during approach, go back to scanning
        if not self.box_detected:
            self.search_state = "SCAN"
            self.get_logger().info('Box lost. Reverting to scan mode.')
        
        return vel_msg
    
    def control_loop(self):
        # Create velocity command
        vel_msg = Twist()
        
        # Execute behavior based on current state
        if self.search_state == "SCAN":
            if self.box_detected:
                self.search_state = "APPROACH"
            else:
                vel_msg = self.execute_scan()
        
        elif self.search_state == "MOVE":
            if self.box_detected:
                self.search_state = "APPROACH"
            else:
                vel_msg = self.execute_move()
        
        elif self.search_state == "APPROACH":
            if not self.box_detected:
                self.search_state = "SCAN"
            else:
                vel_msg = self.execute_approach()
        
        # Publish velocity command
        self.cmd_vel_pub.publish(vel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = EnhancedBoxTracker()
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