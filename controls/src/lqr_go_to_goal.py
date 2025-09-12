#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import math

try:
    from scipy.linalg import solve_continuous_are
except Exception as e:
    raise RuntimeError("scipy is required. Install with: pip install scipy") from e

class LQRGoToGoal(Node):
    def __init__(self):
        super().__init__('lqr_go_to_goal_fixed')

        # Target
        self.target_x = 0.0
        self.target_y = -0.4
        self.target_theta = 0.0

        # Controller parameters
        self.Ts = 0.05  # controller period (used for rate limiting only)
        self.alpha = 0.5  # global scale on computed u to keep outputs tame while tuning

        # Continuous-time simple model:
        # states e = [dist, heading_error], controls u = [v, w]
        # Approximate dynamics: dist_dot = -v  ; heading_dot = -w
        # So A_cont = 0, B_cont = -I  (we'll treat B_cont = I and flip sign in interpretation)
        A_cont = np.zeros((2,2))
        B_cont = np.eye(2)   # choose identity and handle sign by using u = +K @ e

        # LQR weights (tune these)
        q_dist = 5.0
        q_head = 2.0
        self.Q = np.diag([q_dist, q_head])
        r_v = 1.0
        r_w = 0.5
        self.R = np.diag([r_v, r_w])

        # Solve continuous CARE, get continuous-time Kc
        P = solve_continuous_are(A_cont, B_cont, self.Q, self.R)
        Kc = np.linalg.inv(self.R) @ (B_cont.T @ P)  # Kc = R^-1 B^T P

        # We'll use K = Kc (continuous) and apply u = +alpha * (K @ e)
        self.K = Kc

        self.get_logger().info(f"Continuous LQR K:\n{self.K}")

        # Limits
        self.max_v = 0.6
        self.max_w = 1.0

        # thresholds
        self.dist_threshold = 0.05
        self.yaw_threshold = 0.05

        # ROS
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self._last_apply_time = self.get_clock().now()

    def odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        ys = 2.0*(q.w*q.z + q.x*q.y)
        yc = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        yaw = math.atan2(ys, yc)

        dx = self.target_x - px
        dy = self.target_y - py
        dist = math.hypot(dx, dy)

        desired_ang = math.atan2(dy, dx)
        eth = desired_ang - yaw
        eth = math.atan2(math.sin(eth), math.cos(eth))

        # For clarity: ex = distance (positive when target ahead), eth = heading error
        ex = dist

        e = np.array([ex, eth]).reshape((2,1))

        # compute control using continuous K. Note: we use +K*e (not -K*e)
        u = (self.K @ e).flatten()  # u = [uv, uw]
        # apply global scaling alpha to be safe
        v_cmd = float(self.alpha * u[0])
        w_cmd = float(self.alpha * u[1])

        # Because our simplified model assumed dist_dot ≈ -v, a positive v reduces distance.
        # If you observe sign mismatch, flip v_cmd = -v_cmd.

        # saturate
        v_cmd = max(min(v_cmd, self.max_v), -self.max_v)
        w_cmd = max(min(w_cmd, self.max_w), -self.max_w)

        # If close to goal, only rotate to final heading
        if dist < self.dist_threshold:
            if abs(self.target_theta - yaw) > self.yaw_threshold:
                v_cmd = 0.0
                # small yaw controller to settle final heading
                yaw_err = math.atan2(math.sin(self.target_theta - yaw), math.cos(self.target_theta - yaw))
                w_cmd = 0.4 * yaw_err
            else:
                v_cmd = 0.0
                w_cmd = 0.0
                self.get_logger().info("Goal reached!")

        # publish at Ts
        now = self.get_clock().now()
        if (now - self._last_apply_time).nanoseconds * 1e-9 >= self.Ts:
            cmd = Twist()
            cmd.linear.x = v_cmd
            cmd.angular.z = w_cmd
            self.cmd_pub.publish(cmd)
            self._last_apply_time = now

        # debug
        self.get_logger().debug(f"Pose ({px:.2f},{py:.2f}), yaw {yaw:.2f}, dist {dist:.2f}, eth {eth:.2f}")
        self.get_logger().info(f"Computed u (pre-sat): v={v_cmd:.3f}, w={w_cmd:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = LQRGoToGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
