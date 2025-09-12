#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal_controller')
        self.target_x = 0.0
        self.target_y = -0.4

        # PID gains
        self.kp_dist = 1.5
        self.kp_ang = 4.0

        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Orientation to yaw
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

        # Errors
        dx = self.target_x - x
        dy = self.target_y - y
        distance_error = math.sqrt(dx**2 + dy**2)
        desired_yaw = math.atan2(dy, dx)
        heading_error = desired_yaw - yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))  # normalize

        # Control law
        v = self.kp_dist * distance_error
        omega = self.kp_ang * heading_error

        # Saturation (safety)
        v = max(min(v, 0.5), -0.5)
        omega = max(min(omega, 1.0), -1.0)

        cmd = Twist()
        if distance_error > 0.05:  # stop threshold
            cmd.linear.x = v
            cmd.angular.z = omega
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = GoToGoal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
