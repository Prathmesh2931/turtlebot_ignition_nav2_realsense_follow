#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
from std_msgs.msg import Float32
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class ExploringBoxFinder(Node):
    def __init__(self):
        super().__init__('exploring_box_finder')
        
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
        
        # Subscribe to laser scan for navigation
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
        self.box_width = 0
        self.box_height = 0
        
        # State variables
        self.state = "SCANNING"  # SCANNING, MOVING, APPROACHING, ALIGNING
        self.scan_start_time = None
        self.move_start_time = None
        self.move_duration = 0
        self.move_direction = 0  # 0: forward, 1: right, 2: left, 3: backward
        self.rotation_complete = False
        self.move_complete = False
        self.move_distance = 0.5  # meters
        self.scan_count = 0
        self.unexplored_directions = []  # List of angles that haven't been explored yet
        
        # Movement tracking
        self.current_position = [0, 0]  # Estimated position (x, y)
        self.current_orientation = 0.0   # Estimated orientation in radians
        self.visited_positions = []      # List of visited positions
        
        # Box color range in HSV (red box)
        self.lower_color = np.array([0, 100, 100])   # Lower red
        self.upper_color = np.array([10, 255, 255])  # Upper red
        self.lower_color2 = np.array([160, 100, 100])  # Lower red (second range)
        self.upper_color2 = np.array([180, 255, 255])  # Upper red (second range)
        
        # Control loop timer (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Debug variables
        self.last_debug_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.debug_interval = 2.0  # seconds
        
        self.get_logger().info('Exploring Box Finder initialized')
        self.get_logger().info('Starting in SCANNING state')
    
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
        
        # Analyze LIDAR data to find unexplored directions when in SCANNING state
        if self.state == "SCANNING" and not self.unexplored_directions:
            self.find_unexplored_directions()
    
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
                            
                            # If we find the box, switch to APPROACHING mode
                            if self.state in ["SCANNING", "MOVING"]:
                                self.state = "APPROACHING"
                                self.get_logger().info(f'Box detected! Switching to APPROACHING mode. Distance: {self.box_distance:.3f}m')
                        else:
                            self.get_logger().warn('Box detected but no valid depth measurements')
                    
                    # Optional: Draw box on image for visualization (for debugging)
                    # debug_image = self.rgb_image.copy()
                    # cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.putText(debug_image, f'Distance: {self.box_distance:.3f}m', 
                    #             (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # No box detected that meets the area threshold
                    self.box_detected = False
            else:
                # No contours found
                self.box_detected = False
        
        except Exception as e:
            self.get_logger().error(f'Error in box detection: {str(e)}')
    
    def find_unexplored_directions(self):
        """Find directions that haven't been explored yet using LIDAR data"""
        if self.laser_data is None:
            return
        
        # Convert laser scan to numpy array
        ranges = np.array(self.laser_data.ranges)
        angles = np.linspace(self.laser_data.angle_min, self.laser_data.angle_max, len(ranges))
        
        # Replace inf and nan values with a large number (10.0 meters)
        ranges[np.isnan(ranges) | np.isinf(ranges)] = 10.0
        
        # Find directions with large open spaces (range > 2.0m)
        open_directions = angles[ranges > 2.0]
        
        if len(open_directions) > 0:
            # Cluster open directions into sectors
            sectors = []
            current_sector = [open_directions[0]]
            
            for i in range(1, len(open_directions)):
                if open_directions[i] - open_directions[i-1] < 0.2:  # If angles are close
                    current_sector.append(open_directions[i])
                else:
                    # Start a new sector
                    sectors.append(current_sector)
                    current_sector = [open_directions[i]]
            
            # Add the last sector
            if current_sector:
                sectors.append(current_sector)
            
            # Get the middle angle of each sector
            sector_angles = [sum(sector) / len(sector) for sector in sectors]
            
            # Find the widest sectors
            sector_widths = [len(sector) * self.laser_data.angle_increment for sector in sectors]
            
            # Sort sectors by width (largest first)
            sorted_indices = np.argsort(sector_widths)[::-1]
            
            # Get the middle angles of the top 3 widest sectors
            self.unexplored_directions = [sector_angles[i] for i in sorted_indices[:3] if i < len(sector_angles)]
            
            # Log the unexplored directions
            angles_degrees = [math.degrees(angle) for angle in self.unexplored_directions]
            self.get_logger().info(f'Found unexplored directions at: {angles_degrees} degrees')
    
    def is_path_clear(self, direction=0.0, distance=0.5):
        """Check if the path is clear in the specified direction"""
        if self.laser_data is None:
            return False
        
        # Convert laser scan to numpy array
        ranges = np.array(self.laser_data.ranges)
        angles = np.linspace(self.laser_data.angle_min, self.laser_data.angle_max, len(ranges))
        
        # Find angles within 30 degrees of the specified direction
        angle_diff = np.abs(angles - direction)
        # Handle angle wrapping
        angle_diff = np.minimum(angle_diff, 2*math.pi - angle_diff)
        
        front_indices = np.where(angle_diff < math.pi/6)[0]  # Within 30 degrees
        
        if len(front_indices) > 0:
            front_ranges = ranges[front_indices]
            front_ranges = front_ranges[~np.isnan(front_ranges) & ~np.isinf(front_ranges)]
            
            if len(front_ranges) > 0:
                min_range = np.min(front_ranges)
                return min_range > distance
        
        return True  # Default to clear if no valid readings
    
    def add_visited_position(self):
        """Add current position to the list of visited positions"""
        # Only add if it's sufficiently different from existing positions
        for pos in self.visited_positions:
            if math.sqrt((pos[0] - self.current_position[0])**2 + 
                         (pos[1] - self.current_position[1])**2) < 0.3:
                return  # Too close to an existing position
        
        # Add the current position
        self.visited_positions.append(self.current_position.copy())
    
    def update_position_estimate(self, linear_vel, angular_vel, dt):
        """Update the estimated position based on velocities"""
        # Simple odometry calculation
        dx = linear_vel * math.cos(self.current_orientation) * dt
        dy = linear_vel * math.sin(self.current_orientation) * dt
        dtheta = angular_vel * dt
        
        self.current_position[0] += dx
        self.current_position[1] += dy
        self.current_orientation = (self.current_orientation + dtheta) % (2 * math.pi)
    
    def do_scan(self):
        """Execute a 360-degree scan"""
        vel_msg = Twist()
        
        # Initialize scan if it's the first time
        if self.scan_start_time is None:
            self.scan_start_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.rotation_complete = False
            self.get_logger().info('Starting 360-degree scan')
        
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed = current_time - self.scan_start_time
        
        # Complete a full 360-degree rotation in 8 seconds
        if elapsed < 8.0:
            angular_velocity = 2 * math.pi / 8.0  # radians/second
            vel_msg.angular.z = angular_velocity
        else:
            # Scan complete
            self.scan_start_time = None
            self.rotation_complete = True
            self.scan_count += 1
            vel_msg.angular.z = 0.0
            
            # After scanning, transition to MOVING state
            self.state = "MOVING"
            self.get_logger().info('Scan complete. Switching to MOVING state.')
        
        return vel_msg
    
    def do_move(self):
        """Move in the best available direction"""
        vel_msg = Twist()
        
        # Initialize movement if it's the first time
        if self.move_start_time is None:
            self.move_start_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.move_complete = False
            
            # Choose a direction to move
            if self.unexplored_directions:
                # Take the first unexplored direction
                self.move_direction = self.unexplored_directions.pop(0)
            else:
                # If no unexplored directions, choose a random direction
                self.move_direction = np.random.uniform(-math.pi, math.pi)
            
            # Calculate move duration based on distance
            self.move_duration = self.move_distance / 0.2  # at 0.2 m/s
            
            self.get_logger().info(f'Starting to move in direction: {math.degrees(self.move_direction):.1f}° for {self.move_distance:.1f}m')
        
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed = current_time - self.move_start_time
        
        # First, rotate to face the desired direction
        angle_diff = self.move_direction - self.current_orientation
        
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2*math.pi
        while angle_diff < -math.pi:
            angle_diff += 2*math.pi
        
        # Debug output
        now = self.get_clock().now().seconds_nanoseconds()[0]
        if now - self.last_debug_time > self.debug_interval:
            self.last_debug_time = now
            self.get_logger().info(f'Angle diff: {math.degrees(angle_diff):.1f}°, Orientation: {math.degrees(self.current_orientation):.1f}°')
        
        # If not facing the right direction yet
        if abs(angle_diff) > 0.1:
            vel_msg.angular.z = 0.3 if angle_diff > 0 else -0.3
            return vel_msg
        
        # Now facing the right direction, start moving forward
        # Check if path is clear
        if not self.is_path_clear(self.move_direction, self.move_distance):
            # Path not clear, find another direction
            self.move_start_time = None  # Reset to choose a new direction
            self.get_logger().info('Path not clear. Finding another direction.')
            return vel_msg
        
        # Calculate how far we've moved
        if elapsed < self.move_duration:
            # Move forward
            vel_msg.linear.x = 0.2
        else:
            # Movement complete
            self.move_complete = True
            self.move_start_time = None
            
            # Add current position to visited positions
            self.add_visited_position()
            
            # After moving, go back to scanning
            self.state = "SCANNING"
            self.get_logger().info('Move complete. Switching back to SCANNING state.')
        
        return vel_msg
    
    def do_approach(self):
        """Approach the detected box"""
        vel_msg = Twist()
        
        # If box is no longer detected, go back to scanning
        if not self.box_detected:
            self.state = "SCANNING"
            self.get_logger().info('Box lost. Switching back to SCANNING state.')
            return vel_msg
        
        # Calculate distance to maintain (box radius + target distance)
        box_radius = 0.05  # 5 cm radius for a 10cm box
        desired_distance = box_radius + self.target_distance  # 5cm + 2cm = 7cm
        
        # Check if we need to switch to alignment mode
        if abs(self.box_distance - desired_distance) < 0.05:  # Within 5cm of desired distance
            self.state = "ALIGNING"
            self.get_logger().info('Close enough to desired distance. Switching to ALIGNING state.')
            return vel_msg
        
        # Center the box in the image
        if self.rgb_image is not None:
            image_width = self.rgb_image.shape[1]
            image_center_x = image_width // 2
            
            # Calculate error in pixels
            error_x = self.box_position[0] - image_center_x
            
            # Apply angular velocity to center the box
            angular_velocity = -error_x * 0.001
            vel_msg.angular.z = angular_velocity
        
        # If we're further than the desired distance, move toward the box
        if self.box_distance > desired_distance + 0.01:  # 1cm tolerance
            # Calculate speed based on distance (slow down as we get closer)
            speed = min(0.2, max(0.05, (self.box_distance - desired_distance) * 0.5))
            
            # Set linear velocity to approach the box
            vel_msg.linear.x = speed
            
            self.get_logger().info(f'Moving toward box. Distance: {self.box_distance:.3f}m, Target: {desired_distance:.3f}m, Speed: {speed:.3f}m/s')
        elif self.box_distance < desired_distance - 0.01:  # Too close, back up
            # Calculate speed based on distance
            speed = min(0.1, max(0.05, (desired_distance - self.box_distance) * 0.5))
            
            # Set linear velocity to back away from the box
            vel_msg.linear.x = -speed
            
            self.get_logger().info(f'Backing away from box. Distance: {self.box_distance:.3f}m, Target: {desired_distance:.3f}m')
        else:
            # We're at the desired distance
            self.state = "ALIGNING"
            self.get_logger().info(f'At target distance: {self.box_distance:.3f}m. Switching to ALIGNING state.')
        
        return vel_msg
    
    def do_align(self):
        """Align with the box at approximately 90 degrees"""
        vel_msg = Twist()
        
        # If box is no longer detected, go back to scanning
        if not self.box_detected:
            self.state = "SCANNING"
            self.get_logger().info('Box lost during alignment. Switching back to SCANNING state.')
            return vel_msg
        
        # Check if we're roughly aligned with the box (square appearance)
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
                # Apply angular velocity to center the box
                angular_velocity = -error_x * 0.0005
                vel_msg.angular.z = angular_velocity
                self.get_logger().info(f'Centering on box. Error: {error_x} pixels')
                return vel_msg
        
        # If we're not aligned with the box at 90 degrees
        if not is_aligned:
            # If the box appears wider than tall or vice versa
            if aspect_ratio > 1.15:  # Box is wider than tall
                vel_msg.angular.z = 0.1  # Rotate clockwise
                self.get_logger().info(f'Aligning with box: rotating clockwise. Aspect ratio: {aspect_ratio:.2f}')
            elif aspect_ratio < 0.85:  # Box is taller than wide
                vel_msg.angular.z = -0.1  # Rotate counter-clockwise
                self.get_logger().info(f'Aligning with box: rotating counter-clockwise. Aspect ratio: {aspect_ratio:.2f}')
        else:
            # We're aligned enough - human-like behavior doesn't need perfect alignment
            self.get_logger().info('Box alignment achieved! Mission complete.')
            
            # Stop the robot
            vel_msg.linear.x = 0
            vel_msg.angular.z = 0
        
        return vel_msg
    
    def control_loop(self):
        """Main control loop for behavior execution"""
        # Create velocity command
        vel_msg = Twist()
        
        # Execute behavior based on current state
        if self.state == "SCANNING":
            vel_msg = self.do_scan()
        
        elif self.state == "MOVING":
            vel_msg = self.do_move()
        
        elif self.state == "APPROACHING":
            vel_msg = self.do_approach()
        
        elif self.state == "ALIGNING":
            vel_msg = self.do_align()
        
        # Estimate position based on velocities (simple odometry)
        self.update_position_estimate(vel_msg.linear.x, vel_msg.angular.z, 0.1)  # dt = 0.1s
        
        # Publish velocity command
        self.cmd_vel_pub.publish(vel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ExploringBoxFinder()
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