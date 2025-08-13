#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
import time
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class AdaptiveBoxFinder(Node):
    def __init__(self):
        super().__init__('adaptive_box_finder')
        
        # Configure QoS for better reliability
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create CV bridge
        self.bridge = CvBridge()
        
        # Create subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.rgb_callback,
            qos_profile)
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            qos_profile)
        
        # Subscribe to laser scan for environment mapping
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile)
        
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
        self.box_dimensions = [0.1, 0.1, 0.1]  # 10cm box dimensions
        self.box_width = 0
        self.box_height = 0
        self.last_box_detection_time = None
        
        # Environment understanding variables
        self.open_spaces = []  # List of [angle, distance, width] of open spaces
        self.visited_positions = []  # List of [x, y] positions already visited
        self.current_position = [0, 0]  # Current estimated position
        self.current_orientation = 0.0  # Current estimated orientation
        self.environment_map = np.zeros((100, 100), dtype=np.uint8)  # Simple occupancy grid
        self.map_resolution = 0.1  # meters per cell
        self.map_origin = [50, 50]  # Center of the map
        
        # Behavioral states
        self.behavior_state = "EXPLORE"  # EXPLORE, SEARCH, APPROACH, ALIGN
        self.exploration_start_time = None
        self.current_goal = None  # [x, y] position to move to
        self.rotation_start_time = None
        self.search_timeout = 60.0  # seconds before changing search area
        self.last_behavior_change_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.alignment_tolerance = 15.0  # degrees (more relaxed alignment)
        self.stop_at_close_enough = True  # Stop when reasonably close and aligned
        self.found_box_once = False  # Flag to know if we've seen the box before
        
        # Box color range in HSV (red box)
        # Adjust these values based on your actual box color
        self.lower_color = np.array([0, 100, 100])   # Lower red
        self.upper_color = np.array([10, 255, 255])  # Upper red
        self.lower_color2 = np.array([160, 100, 100])  # Lower red (second range)
        self.upper_color2 = np.array([180, 255, 255])  # Upper red (second range)
        
        # Control loop timer (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Adaptive box finder initialized')
    
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
            
            # Analyze depth image to find open spaces
            if self.behavior_state == "EXPLORE" and not self.box_detected:
                self.find_open_spaces_from_depth()
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')
    
    def scan_callback(self, msg):
        self.laser_data = msg
        
        # Update environment map when in explore mode
        if self.behavior_state == "EXPLORE":
            self.update_environment_map()
    
    def detect_box(self):
        if self.rgb_image is None:
            return
        
        try:
            # Convert BGR to HSV
            hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
            
            # Threshold the HSV image to get only red colors
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
                    self.last_box_detection_time = self.get_clock().now().seconds_nanoseconds()[0]
                    self.found_box_once = True
                    
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
                            
                            # If we find the box, switch to APPROACH mode
                            if self.behavior_state in ["EXPLORE", "SEARCH"]:
                                self.behavior_state = "APPROACH"
                                self.get_logger().info(f'Box detected! Switching to APPROACH mode. Distance: {self.box_distance:.3f}m')
                        else:
                            self.get_logger().warn('Box detected but no valid depth measurements')
                    
                    # Optional: Draw box on image for visualization
                    debug_image = self.rgb_image.copy()
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(debug_image, f'Distance: {self.box_distance:.3f}m', 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # If you want to publish this image for debugging, add a publisher here
                else:
                    # No box detected that meets the area threshold
                    self.handle_box_not_detected()
            else:
                # No contours found
                self.handle_box_not_detected()
        
        except Exception as e:
            self.get_logger().error(f'Error in box detection: {str(e)}')
    
    def handle_box_not_detected(self):
        """Handle the case when the box is not detected"""
        # Only consider the box lost if it was previously detected
        if self.box_detected:
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            
            # If box hasn't been seen for 2 seconds, consider it lost
            if self.last_box_detection_time and (current_time - self.last_box_detection_time) > 2.0:
                self.box_detected = False
                
                # If we were approaching or aligning, go back to search mode
                if self.behavior_state in ["APPROACH", "ALIGN"]:
                    self.behavior_state = "SEARCH"
                    self.get_logger().info('Box lost. Switching to SEARCH mode.')
    
    def find_open_spaces_from_depth(self):
        """Analyze depth image to find open spaces"""
        if self.depth_image is None:
            return
        
        try:
            # Create a binary map of obstacles vs. free space
            # Pixels with distance > 1.0m are considered open space
            free_space = np.zeros_like(self.depth_image, dtype=np.uint8)
            free_space[self.depth_image > 1.0] = 255
            free_space[np.isnan(self.depth_image) | np.isinf(self.depth_image)] = 0
            
            # Apply morphological operations to clean up the map
            kernel = np.ones((5, 5), np.uint8)
            free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel)
            
            # Find contours of open spaces
            contours, _ = cv2.findContours(free_space, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest open space
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 5000:  # Only consider large open spaces
                    # Get center of the open space
                    M = cv2.moments(largest_contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Convert to angle from center of image
                        h, w = self.depth_image.shape
                        angle = math.atan2(cx - w/2, h/2 - cy)  # Angle in radians
                        
                        # Get distance to this point
                        distance = self.depth_image[cy, cx] if not np.isnan(self.depth_image[cy, cx]) and not np.isinf(self.depth_image[cy, cx]) else 2.0
                        
                        # Update open spaces list
                        self.open_spaces.append([angle, distance, area])
                        
                        # Sort open spaces by area (largest first)
                        self.open_spaces.sort(key=lambda x: x[2], reverse=True)
                        
                        # Keep only the top 5 open spaces
                        self.open_spaces = self.open_spaces[:5]
                        
                        self.get_logger().debug(f'Found open space at angle: {math.degrees(angle):.1f}°, distance: {distance:.2f}m, area: {area}')
        except Exception as e:
            self.get_logger().error(f'Error finding open spaces: {str(e)}')
    
    def update_environment_map(self):
        """Update the environment map using laser scan data"""
        if self.laser_data is None:
            return
        
        try:
            # Convert laser scan to points in the map
            angle = self.laser_data.angle_min
            for r in self.laser_data.ranges:
                if not np.isnan(r) and not np.isinf(r):
                    # Convert polar to Cartesian coordinates
                    x = r * math.cos(angle)
                    y = r * math.sin(angle)
                    
                    # Convert to map coordinates
                    map_x = int(self.map_origin[0] + x / self.map_resolution)
                    map_y = int(self.map_origin[1] + y / self.map_resolution)
                    
                    # Check if coordinates are within map bounds
                    if 0 <= map_x < self.environment_map.shape[1] and 0 <= map_y < self.environment_map.shape[0]:
                        # Mark as obstacle
                        self.environment_map[map_y, map_x] = 255
                
                angle += self.laser_data.angle_increment
        
        except Exception as e:
            self.get_logger().error(f'Error updating environment map: {str(e)}')
    
    def find_best_exploration_direction(self):
        """Find the best direction to explore"""
        # If we have detected open spaces from the depth image
        if self.open_spaces:
            # Return the angle to the largest open space
            return self.open_spaces[0][0]
        
        # If we have laser data, find the longest distance
        elif self.laser_data is not None:
            max_distance = 0
            max_angle = 0
            angle = self.laser_data.angle_min
            
            for i, r in enumerate(self.laser_data.ranges):
                if not np.isnan(r) and not np.isinf(r) and r > max_distance:
                    max_distance = r
                    max_angle = angle
                
                angle += self.laser_data.angle_increment
            
            return max_angle
        
        # Default: turn randomly if no sensor data available
        return np.random.uniform(-math.pi/4, math.pi/4)
    
    def explore_environment(self):
        """Explore the environment in search of the box"""
        vel_msg = Twist()
        
        # Initialize exploration if it's the first time
        if self.exploration_start_time is None:
            self.exploration_start_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.current_goal = None
            
            # Determine best direction to explore
            explore_angle = self.find_best_exploration_direction()
            
            # Set as current goal
            self.current_goal = explore_angle
            self.get_logger().info(f'Starting exploration in direction: {math.degrees(explore_angle):.1f}°')
        
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        
        # Check if we should find a new exploration direction
        if (current_time - self.exploration_start_time) > 10.0:  # Every 10 seconds
            # Reset exploration timer
            self.exploration_start_time = current_time
            
            # Find new direction
            explore_angle = self.find_best_exploration_direction()
            self.current_goal = explore_angle
            
            self.get_logger().info(f'New exploration direction: {math.degrees(explore_angle):.1f}°')
        
        # If we have a goal direction
        if self.current_goal is not None:
            # First turn towards the goal direction
            angle_diff = self.current_goal  # Current goal is the angle to turn
            
            if abs(angle_diff) > 0.1:  # If we're not facing the right direction
                # Turn towards the direction
                vel_msg.angular.z = 0.3 if angle_diff > 0 else -0.3
            else:
                # We're facing the right direction, move forward
                vel_msg.linear.x = 0.2
                
                # Occasionally look around while moving
                if np.random.random() < 0.2:  # 20% chance to look around
                    vel_msg.angular.z = np.random.uniform(-0.2, 0.2)
        else:
            # No goal, just spin slowly to look around
            vel_msg.angular.z = 0.2
        
        # Check for obstacles directly ahead
        if self.laser_data is not None:
            # Check the front 30 degrees
            front_indices = np.where(
                (np.array(range(len(self.laser_data.ranges))) * self.laser_data.angle_increment + 
                 self.laser_data.angle_min > -math.pi/12) & 
                (np.array(range(len(self.laser_data.ranges))) * self.laser_data.angle_increment + 
                 self.laser_data.angle_min < math.pi/12)
            )[0]
            
            if len(front_indices) > 0:
                front_distances = np.array(self.laser_data.ranges)[front_indices]
                min_front_distance = np.min(front_distances[~np.isnan(front_distances) & ~np.isinf(front_distances)]) if len(front_distances) > 0 else float('inf')
                
                # If obstacle is too close and we're moving forward
                if min_front_distance < 0.5 and vel_msg.linear.x > 0:
                    vel_msg.linear.x = 0  # Stop forward motion
                    vel_msg.angular.z = 0.5  # Turn to avoid obstacle
                    self.get_logger().info(f'Obstacle detected at {min_front_distance:.2f}m, turning to avoid')
        
        # Periodically switch to search mode
        if self.found_box_once and (current_time - self.last_behavior_change_time) > 30.0:
            self.behavior_state = "SEARCH"
            self.last_behavior_change_time = current_time
            self.get_logger().info('Switching to focused SEARCH mode')
            vel_msg.linear.x = 0
            vel_msg.angular.z = 0
        
        return vel_msg
    
    def search_for_box(self):
        """Perform a more focused search for the box"""
        vel_msg = Twist()
        
        # Initialize rotation if it's the first time
        if self.rotation_start_time is None:
            self.rotation_start_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.get_logger().info('Starting focused search rotation')
        
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed = current_time - self.rotation_start_time
        
        # Rotate 360 degrees over 10 seconds
        if elapsed < 10.0:
            vel_msg.angular.z = 2 * math.pi / 10.0  # Full circle in 10 seconds
        else:
            # After full rotation, move to a new area
            self.rotation_start_time = None
            
            # Switch back to exploration if we haven't found the box
            if not self.box_detected:
                self.behavior_state = "EXPLORE"
                self.exploration_start_time = None
                self.get_logger().info('Search complete, box not found. Switching back to EXPLORE mode')
        
        # If search takes too long without finding the box
        if elapsed > self.search_timeout:
            self.behavior_state = "EXPLORE"
            self.exploration_start_time = None
            self.get_logger().info('Search timeout reached. Switching back to EXPLORE mode')
        
        return vel_msg
    
    def approach_box(self):
        """Approach the detected box"""
        vel_msg = Twist()
        
        # Calculate distance to maintain (box radius + target distance)
        box_radius = 0.05  # 5 cm radius for a 10cm box
        desired_distance = box_radius + self.target_distance  # 5cm + 2cm = 7cm
        
        # Check if we need to switch to alignment mode
        if abs(self.box_distance - desired_distance) < 0.05:  # Within 5cm of desired distance
            self.behavior_state = "ALIGN"
            self.get_logger().info('Close enough to desired distance. Switching to ALIGN mode')
            return vel_msg
        
        # If we're further than the desired distance, move toward the box
        if self.box_distance > desired_distance:
            # Calculate speed based on distance (slow down as we get closer)
            # More human-like: start slower, then speed up, then slow down again
            dist_diff = self.box_distance - desired_distance
            
            # Sigmoidal speed profile
            speed = 0.3 / (1 + math.exp(-5 * (dist_diff - 0.5)))
            speed = max(0.05, min(0.2, speed))  # Clamp between 0.05 and 0.2 m/s
            
            # Set linear velocity to approach the box
            vel_msg.linear.x = speed
            
            # Calculate angular velocity to center the box in the image
            if self.rgb_image is not None:
                image_width = self.rgb_image.shape[1]
                image_center_x = image_width // 2
                
                # Calculate error in pixels
                error_x = self.box_position[0] - image_center_x
                
                # Human-like steering: more aggressive when error is large
                angular_velocity = -error_x * 0.001 * (1 + abs(error_x) / 100)
                vel_msg.angular.z = angular_velocity
            
            self.get_logger().info(f'Moving toward box. Distance: {self.box_distance:.3f}m, Target: {desired_distance:.3f}m, Speed: {speed:.3f}m/s')
        elif self.box_distance < desired_distance - 0.03:  # Too close, back up (with more tolerance)
            # Calculate speed based on distance
            speed = min(0.1, max(0.05, (desired_distance - self.box_distance) * 0.3))
            
            # Set linear velocity to back away from the box
            vel_msg.linear.x = -speed
            
            self.get_logger().info(f'Backing away from box. Distance: {self.box_distance:.3f}m, Target: {desired_distance:.3f}m')
        else:
            # We're close enough to the desired distance
            self.behavior_state = "ALIGN"
            self.get_logger().info(f'At target distance: {self.box_distance:.3f}m. Switching to ALIGN mode')
        
        return vel_msg
    
    def align_with_box(self):
        """Align with the box at approximately 90 degrees"""
        vel_msg = Twist()
        
        # If box is no longer detected, go back to search
        if not self.box_detected:
            self.behavior_state = "SEARCH"
            self.get_logger().info('Box lost during alignment. Switching to SEARCH mode')
            return vel_msg
        
        # Check if we're roughly aligned with the box
        # For a square box, we look at aspect ratio
        aspect_ratio = float(self.box_width) / self.box_height if self.box_height > 0 else 1.0
        
        # More relaxed alignment check (15% tolerance)
        is_aligned = 0.85 < aspect_ratio < 1.15
        
        # Center the box in the image
        if self.rgb_image is not None:
            image_width = self.rgb_image.shape[1]
            image_center_x = image_width // 2
            
            # Calculate error in pixels
            error_x = self.box_position[0] - image_center_x
            is_centered = abs(error_x) < 30  # More relaxed centering (30 pixels)
            
            if not is_centered:
                # Calculate angular velocity to center
                angular_velocity = -error_x * 0.0005
                vel_msg.angular.z = angular_velocity
                self.get_logger().info(f'Centering on box. Error: {error_x} pixels')
                return vel_msg
        
        # If we're not aligned with the box at 90 degrees
        if not is_aligned:
            # If the box appears wider than tall or vice versa
            if aspect_ratio > 1.15:  # Box is wider than tall
                vel_msg.angular.z = 0.1  # Rotate clockwise
                self.get_logger().info('Aligning with box: rotating clockwise')
            elif aspect_ratio < 0.85:  # Box is taller than wide
                vel_msg.angular.z = -0.1  # Rotate counter-clockwise
                self.get_logger().info('Aligning with box: rotating counter-clockwise')
        else:
            # We're aligned enough - human-like behavior doesn't need perfect alignment
            self.get_logger().info('Box alignment achieved! Mission complete.')
            
            # Stop the robot
            vel_msg.linear.x = 0
            vel_msg.angular.z = 0
            
            # Optional: Add a flag to indicate success
            self.stop_at_close_enough = False  # Prevent further movement
        
        return vel_msg
    
    def control_loop(self):
        """Main control loop for behavior execution"""
        # Create velocity command
        vel_msg = Twist()
        
        # Execute behavior based on current state
        if self.behavior_state == "EXPLORE":
            vel_msg = self.explore_environment()
        
        elif self.behavior_state == "SEARCH":
            vel_msg = self.search_for_box()
        
        elif self.behavior_state == "APPROACH":
            vel_msg = self.approach_box()
        
        elif self.behavior_state == "ALIGN":
            vel_msg = self.align_with_box()
        
        # If we're supposed to stop when close enough
        if not self.stop_at_close_enough and self.behavior_state == "ALIGN":
            vel_msg.linear.x = 0
            vel_msg.angular.z = 0
        
        # Publish velocity command
        self.cmd_vel_pub.publish(vel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveBoxFinder()
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