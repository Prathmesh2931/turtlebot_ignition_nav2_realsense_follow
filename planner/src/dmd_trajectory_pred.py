#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.linalg import svd
from collections import deque
import math

class DMDTrajectoryPredictor(Node):
    def __init__(self):
        super().__init__('dmd_trajectory_predictor')
        
        # DMD Parameters
        self.window_size = 50  # Number of historical states for DMD
        self.prediction_horizon = 10  # Future steps to predict
        self.state_history = deque(maxlen=self.window_size)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.rgb_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.predicted_path_pub = self.create_publisher(Point, '/predicted_target', 10)
        
        # State variables
        self.rgb_image = None
        self.depth_image = None
        self.current_pose = None
        self.target_state = np.array([0, 0, 0, 0])  # [x, y, vx, vy]
        self.robot_state = np.array([0, 0, 0, 0])   # [x, y, vx, vy]
        
        # DMD matrices
        self.A_dmd = None
        self.modes = None
        self.eigenvalues = None
        
        # Control parameters
        self.target_distance = 1.0  # meters
        self.max_linear_vel = 0.5
        self.max_angular_vel = 1.0
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('DMD Trajectory Predictor initialized')
    
    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.detect_and_track_target()
        except Exception as e:
            self.get_logger().error(f'RGB processing error: {str(e)}')
    
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f'Depth processing error: {str(e)}')
    
    def scan_callback(self, msg):
        # Use LIDAR for obstacle avoidance in DMD predictions
        self.laser_data = msg
    
    def odom_callback(self, msg):
        # Update robot state
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        self.robot_state = np.array([pos.x, pos.y, vel.x, vel.y])
        self.current_pose = msg.pose.pose
    
    def detect_and_track_target(self):
        if self.rgb_image is None or self.depth_image is None:
            return
        
        # Enhanced target detection using depth information
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        
        # Red box detection (multiple ranges for better detection)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 500:  # Minimum area threshold
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x, center_y = x + w//2, y + h//2
                
                # Get 3D position using depth data
                if 0 <= center_y < self.depth_image.shape[0] and 0 <= center_x < self.depth_image.shape[1]:
                    depth = self.depth_image[center_y, center_x]
                    
                    if not np.isnan(depth) and depth > 0:
                        # Convert pixel coordinates to world coordinates
                        target_x, target_y = self.pixel_to_world(center_x, center_y, depth)
                        
                        # Estimate velocity using finite differences
                        if len(self.state_history) > 0:
                            prev_state = self.state_history[-1]
                            dt = 0.1  # Control loop period
                            vx = (target_x - prev_state[0]) / dt
                            vy = (target_y - prev_state[1]) / dt
                        else:
                            vx, vy = 0.0, 0.0
                        
                        # Update target state
                        new_state = np.array([target_x, target_y, vx, vy])
                        self.target_state = new_state
                        self.state_history.append(new_state)
                        
                        # Update DMD model
                        self.update_dmd_model()
    
    def pixel_to_world(self, u, v, depth):
        # Camera intrinsics (typical RealSense values - adjust as needed)
        fx, fy = 554.254691191187, 554.254691191187
        cx, cy = 320.5, 240.5
        
        # Convert to camera coordinates
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        
        # Transform to robot base frame (assuming camera is at robot center)
        # Add your specific camera-to-base transform here
        x_world = x_cam
        y_world = y_cam
        
        return x_world, y_world
    
    def update_dmd_model(self):
        if len(self.state_history) < self.window_size:
            return
        
        # Create data matrices for DMD
        states = np.array(list(self.state_history))
        X = states[:-1].T  # State matrix (n x m-1)
        Y = states[1:].T   # Next state matrix (n x m-1)
        
        # Perform DMD
        try:
            # SVD of X
            U, s, Vt = svd(X, full_matrices=False)
            
            # Truncate for numerical stability
            r = min(len(s), 10)  # Keep top 10 modes
            U_r = U[:, :r]
            s_r = s[:r]
            V_r = Vt[:r, :].T
            
            # Build Atilde
            Atilde = U_r.T @ Y @ V_r @ np.diag(1/s_r)
            
            # Eigendecomposition of Atilde
            eigenvals, W = np.linalg.eig(Atilde)
            
            # DMD modes
            Phi = Y @ V_r @ np.diag(1/s_r) @ W
            
            # Store DMD model
            self.A_dmd = Atilde
            self.modes = Phi
            self.eigenvalues = eigenvals
            
        except Exception as e:
            self.get_logger().warn(f'DMD update failed: {str(e)}')
    
    def predict_future_states(self):
        if self.A_dmd is None or len(self.state_history) == 0:
            return [self.target_state]
        
        try:
            # Initial condition (current state)
            x0 = self.state_history[-1]
            
            # Project onto DMD modes
            if self.modes is not None:
                b = np.linalg.pinv(self.modes) @ x0
                
                # Predict future states
                predictions = []
                for k in range(1, self.prediction_horizon + 1):
                    # Evolve dynamics: x_k = Phi * Lambda^k * b
                    x_pred = self.modes @ (np.diag(self.eigenvalues**k) @ b)
                    predictions.append(x_pred.real)  # Take real part
                
                return predictions
            
        except Exception as e:
            self.get_logger().warn(f'Prediction failed: {str(e)}')
        
        return [self.target_state]
    
    def control_loop(self):
        if len(self.state_history) == 0:
            return
        
        # Predict future target positions
        future_states = self.predict_future_states()
        
        if future_states:
            # Use first predicted state for control
            predicted_target = future_states[0]
            
            # Publish predicted target
            target_msg = Point()
            target_msg.x = float(predicted_target[0])
            target_msg.y = float(predicted_target[1])
            target_msg.z = 0.0
            self.predicted_path_pub.publish(target_msg)
            
            # Compute control command
            cmd_vel = self.compute_control_command(predicted_target)
            self.cmd_vel_pub.publish(cmd_vel)
    
    def compute_control_command(self, target_pos):
        cmd_vel = Twist()
        
        if self.current_pose is None:
            return cmd_vel
        
        # Get current robot position
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        
        # Compute distance and angle to predicted target
        dx = target_pos[0] - robot_x
        dy = target_pos[1] - robot_y
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        
        # Get current robot orientation
        from tf_transformations import euler_from_quaternion
        orientation = self.current_pose.orientation
        _, _, current_yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        
        # Compute angle error
        angle_error = target_angle - current_yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))  # Normalize
        
        # Control law with predictive adjustment
        if distance > self.target_distance:
            # Move towards target
            cmd_vel.linear.x = min(self.max_linear_vel, 0.5 * distance)
            cmd_vel.angular.z = min(self.max_angular_vel, 2.0 * angle_error)
        else:
            # Maintain distance
            cmd_vel.linear.x = 0.1
            cmd_vel.angular.z = min(self.max_angular_vel, 1.0 * angle_error)
        
        # Add obstacle avoidance using LIDAR
        if hasattr(self, 'laser_data') and self.laser_data is not None:
            cmd_vel = self.apply_obstacle_avoidance(cmd_vel)
        
        return cmd_vel
    
    def apply_obstacle_avoidance(self, cmd_vel):
        # Simple obstacle avoidance using laser scan
        if self.laser_data is None:
            return cmd_vel
        
        ranges = np.array(self.laser_data.ranges)
        angles = np.linspace(self.laser_data.angle_min, self.laser_data.angle_max, len(ranges))
        
        # Check front region
        front_indices = np.where(np.abs(angles) < np.pi/6)[0]
        if len(front_indices) > 0:
            front_ranges = ranges[front_indices]
            front_ranges = front_ranges[~np.isnan(front_ranges) & ~np.isinf(front_ranges)]
            
            if len(front_ranges) > 0 and np.min(front_ranges) < 0.8:
                # Obstacle detected, reduce speed and add avoidance
                cmd_vel.linear.x *= 0.3
                
                # Turn away from closest obstacle
                min_idx = np.argmin(front_ranges)
                obstacle_angle = angles[front_indices[min_idx]]
                        
                # Turn opposite direction
                if obstacle_angle > 0:
                    cmd_vel.angular.z -= 0.5
                else:
                    cmd_vel.angular.z += 0.5
        
        return cmd_vel


def main(args=None):
    rclpy.init(args=args)
    node = DMDTrajectoryPredictor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()