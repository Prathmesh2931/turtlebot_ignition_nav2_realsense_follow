#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import math
from scipy.linalg import solve_continuous_are

class LQRWaypointFollower(Node):
    def __init__(self):
        super().__init__('lqr_waypoint_follower')

        # Waypoints (x, y, theta)
        self.waypoints = [
            (0.5, 0.5, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, math.pi/2),
        ]
        self.current_idx = 0

        # Controller parameters
        self.Ts = 0.05
        self.alpha = 0.5

        # Continuous system model
        A_cont = np.zeros((2,2))
        B_cont = np.eye(2)

        # LQR weights
        q_dist = 5.0
        q_head = 2.0
        self.Q = np.diag([q_dist, q_head])
        r_v = 1.0
        r_w = 0.5
        self.R = np.diag([r_v, r_w])

        # Solve CARE
        P = solve_continuous_are(A_cont, B_cont, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ (B_cont.T @ P)
        self.get_logger().info(f"LQR Gain K:\n{self.K}")

        # Limits (reduced to avoid spiraling)
        self.max_v = 0.3
        self.max_w = 0.5

        # Thresholds
        self.dist_threshold = 0.1
        self.yaw_threshold = 0.1

        # ROS comms
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self._last_apply_time = self.get_clock().now()

    def odom_cb(self, msg: Odometry):
        if self.current_idx >= len(self.waypoints):
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            self.get_logger().info("All waypoints reached ✅")
            return

        # Current pose
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        ys = 2.0*(q.w*q.z + q.x*q.y)
        yc = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        yaw = math.atan2(ys, yc)

        # Current target
        tx, ty, ttheta = self.waypoints[self.current_idx]

        # Errors
        dx = tx - px
        dy = ty - py
        dist = math.hypot(dx, dy)
        desired_ang = math.atan2(dy, dx)
        eth = desired_ang - yaw
        eth = math.atan2(math.sin(eth), math.cos(eth))  # normalize
        ex = dist

        e = np.array([ex, eth]).reshape((2,1))
        u = (self.K @ e).flatten()

        # Flip sign to match real dynamics
        v_cmd = float(self.alpha * u[0])
        w_cmd = float(self.alpha * u[1])

        # Rotate first if heading error is large
        if abs(eth) > 0.3:
            v_cmd = 0.0

        # Saturation
        v_cmd = max(min(v_cmd, self.max_v), -self.max_v)
        w_cmd = max(min(w_cmd, self.max_w), -self.max_w)

        # Goal check
        if dist < self.dist_threshold and abs(ttheta - yaw) < self.yaw_threshold:
            self.get_logger().info(f"Reached waypoint {self.current_idx}: ({tx:.2f},{ty:.2f})")
            self.current_idx += 1
            return

        # Publish
        now = self.get_clock().now()
        if (now - self._last_apply_time).nanoseconds * 1e-9 >= self.Ts:
            cmd = Twist()
            cmd.linear.x = v_cmd
            cmd.angular.z = w_cmd
            self.cmd_pub.publish(cmd)
            self._last_apply_time = now

        self.get_logger().info(
            f"u: v={v_cmd:.3f}, w={w_cmd:.3f} | Target=({tx:.2f},{ty:.2f}), dist={dist:.2f}, eth={eth:.2f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = LQRWaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
