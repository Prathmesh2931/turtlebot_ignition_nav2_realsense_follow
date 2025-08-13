#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class RedBoxFollower(Node):
    def __init__(self):
        super().__init__('red_box_follower')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publisher for robot movement commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers for camera data
        self.rgb_image_sub = self.create_subscription(
            Image, 
            '/camera/rgb/image_raw', 
            self.rgb_image_callback, 
            10
        )
        
        self.depth_image_sub = self.create_subscription(
            Image, 
            '/camera/depth/image_raw', 
            self.depth_image_callback, 
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )
        
        # State variables
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.image_width = 640
        self.image_height = 480
        
        # Red box detection parameters
        self.red_lower = np.array([0, 100, 100])    # Lower HSV threshold for red
        self.red_upper = np.array([10, 255, 255])   # Upper HSV threshold for red
        self.red_lower2 = np.array([160, 100, 100]) # Second range for red (wraps around)
        self.red_upper2 = np.array([180, 255, 255])
        
        # Control parameters
        self.target_distance = 1.5  # Desired distance to box in meters
        self.distance_tolerance = 0.3  # Tolerance for distance control
        self.angular_gain = 0.8     # Proportional gain for angular control
        self.linear_gain = 0.5      # Proportional gain for linear control
        self.max_linear_vel = 0.3   # Maximum linear velocity
        self.max_angular_vel = 0.8  # Maximum angular velocity
        
        # Create timer for main control loop
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        
        self.get_logger().info('Red Box Follower node initialized')

    def camera_info_callback(self, msg):
        """Store camera info for coordinate transformations"""
        self.camera_info = msg
        self.image_width = msg.width
        self.image_height = msg.height

    def rgb_image_callback(self, msg):
        """Process RGB image for red box detection"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting RGB image: {e}')

    def depth_image_callback(self, msg):
        """Process depth image"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')

    def detect_red_box(self, rgb_image):
        """
        Detect red box in RGB image
        Returns: (center_x, center_y, box_area) or None if not found
        """
        if rgb_image is None:
            return None
            
        # Convert BGR to HSV
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for red color (two ranges due to hue wrapping)
        mask1 = cv2.inRange(hsv, self.red_lower, self.red_upper)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask = mask1 + mask2
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (assumed to be the red box)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Filter out small detections
        if area < 500:  # Minimum area threshold
            return None
        
        # Calculate center of the bounding box
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
            return center_x, center_y, area
        
        return None

    def get_distance_to_box(self, center_x, center_y):
        """
        Get distance to the red box using depth information
        """
        if self.depth_image is None:
            return None
            
        # Ensure coordinates are within image bounds
        if (0 <= center_x < self.depth_image.shape[1] and 
            0 <= center_y < self.depth_image.shape[0]):
            
            # Get depth value at the box center (average around center for stability)
            region_size = 10
            y_min = max(0, center_y - region_size)
            y_max = min(self.depth_image.shape[0], center_y + region_size)
            x_min = max(0, center_x - region_size)
            x_max = min(self.depth_image.shape[1], center_x + region_size)
            
            depth_region = self.depth_image[y_min:y_max, x_min:x_max]
            
            # Filter out invalid depth values (0 or inf)
            valid_depths = depth_region[(depth_region > 0) & (depth_region < 10.0)]
            
            if len(valid_depths) > 0:
                return np.median(valid_depths)
        
        return None

    def control_loop(self):
        """Main control loop for following the red box"""
        if self.rgb_image is None:
            return
            
        # Detect red box
        detection = self.detect_red_box(self.rgb_image)
        
        # Initialize command
        cmd = Twist()
        
        if detection is None:
            # No red box detected - stop and search
            cmd.linear.x = 0.0
            cmd.angular.z = 0.1  # Slow rotation to search for box
            self.get_logger().info('No red box detected - searching...')
        else:
            center_x, center_y, area = detection
            
            # Calculate angular error (how far left/right the box is)
            image_center_x = self.image_width // 2
            angular_error = (center_x - image_center_x) / (self.image_width // 2)
            
            # Calculate angular velocity (positive = turn left)
            cmd.angular.z = -self.angular_gain * angular_error
            cmd.angular.z = max(-self.max_angular_vel, 
                              min(self.max_angular_vel, cmd.angular.z))
            
            # Get distance to box
            distance = self.get_distance_to_box(center_x, center_y)
            
            if distance is not None:
                # Calculate distance error
                distance_error = distance - self.target_distance
                
                # Calculate linear velocity
                if abs(distance_error) > self.distance_tolerance:
                    cmd.linear.x = self.linear_gain * distance_error
                    cmd.linear.x = max(-self.max_linear_vel, 
                                     min(self.max_linear_vel, cmd.linear.x))
                else:
                    cmd.linear.x = 0.0
                
                self.get_logger().info(
                    f'Box detected at ({center_x}, {center_y}), '
                    f'distance: {distance:.2f}m, '
                    f'cmd: linear={cmd.linear.x:.2f}, angular={cmd.angular.z:.2f}'
                )
            else:
                # Can't determine distance, move slowly towards box
                cmd.linear.x = 0.1
                self.get_logger().info(
                    f'Box detected but no depth info, '
                    f'cmd: linear={cmd.linear.x:.2f}, angular={cmd.angular.z:.2f}'
                )
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    
    red_box_follower = RedBoxFollower()
    
    try:
        rclpy.spin(red_box_follower)
    except KeyboardInterrupt:
        pass
    finally:
        red_box_follower.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()