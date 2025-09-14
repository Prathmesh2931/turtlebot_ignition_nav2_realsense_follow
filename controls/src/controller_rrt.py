#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
import math, time

class ControllerNode(Node):
    def __init__(self):
        super().__init__('path_follower_node')

        # ---------- Parameters (tweak these) ----------
        self.declare_parameter('follow_distance_thresh', 0.18)   # when path[0] considered reached
        self.declare_parameter('lookahead_dist', 0.6)            # choose waypoint ~this far ahead
        self.declare_parameter('max_lookahead_idx', 6)          # max indexes to skip forward
        self.declare_parameter('stop_dist', 0.20)               # immediate stop threshold (very close)
        self.declare_parameter('immediate_stop_front_angle_deg', 25.0)  # front cone angle for immediate stop
        self.declare_parameter('avoidance_radius', 1.0)         # radius to consider points for avoidance
        self.declare_parameter('avoidance_influence', 0.5)     # strong repulsion within this distance
        self.declare_parameter('avoidance_strength', 1.2)      # scaling of repulsive vector
        self.declare_parameter('replan_threshold_cycles', 12)
        self.declare_parameter('replan_request_cooldown', 2.0)

        # escape maneuver params
        self.declare_parameter('escape_reverse_time', 0.35)  # seconds
        self.declare_parameter('escape_rotate_time', 0.6)    # seconds

        # velocity limits
        self.declare_parameter('max_linear_speed', 0.35)
        self.declare_parameter('max_angular_speed', 1.0)

        # load params
        self.follow_distance_thresh = self.get_parameter('follow_distance_thresh').value
        self.lookahead_dist = self.get_parameter('lookahead_dist').value
        self.max_lookahead_idx = int(self.get_parameter('max_lookahead_idx').value)
        self.stop_dist = self.get_parameter('stop_dist').value
        self.immediate_stop_front_angle = math.radians(self.get_parameter('immediate_stop_front_angle_deg').value)
        self.avoidance_radius = self.get_parameter('avoidance_radius').value
        self.avoidance_influence = self.get_parameter('avoidance_influence').value
        self.avoidance_strength = self.get_parameter('avoidance_strength').value
        self.replan_threshold_cycles = int(self.get_parameter('replan_threshold_cycles').value)
        self.replan_request_cooldown = self.get_parameter('replan_request_cooldown').value

        self.escape_reverse_time = self.get_parameter('escape_reverse_time').value
        self.escape_rotate_time = self.get_parameter('escape_rotate_time').value

        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value

        # ---------- State ----------
        self.path = []            # list of (x,y) in odom frame
        self.current_pose = None  # geometry_msgs/Pose
        self.latest_scan = None   # sensor_msgs/LaserScan
        self.blocked_counter = 0
        self.prev_goal_dist = None

        self.last_replan_time = 0.0
        self.in_escape = False
        self.escape_start_time = None
        self.escape_phase = None  # 'reverse' or 'rotate'

        # ---------- Publishers & Subscribers ----------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.replan_pub = self.create_publisher(Bool, '/replan_request', 10)

        self.create_subscription(Path, '/rrt_path', self.path_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 20)

        self.create_timer(0.1, self.control_loop)

        self.get_logger().info('ControllerNode initialized (vector-based local avoidance + lookahead)')

    # ---------- Callbacks ----------
    def path_cb(self, msg: Path):
        pts = []
        for p in msg.poses:
            pts.append((p.pose.position.x, p.pose.position.y))
        self.path = pts
        self.get_logger().info(f'Controller: received path with {len(self.path)} pts')
        # reset progress trackers
        self.blocked_counter = 0
        self.prev_goal_dist = None

    def odom_cb(self, msg: Odometry):
        self.current_pose = msg.pose.pose

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    # ---------- Control loop ----------
    def control_loop(self):
        if not self.path or self.current_pose is None:
            self.publish_zero()
            return

        # prune reached points (path[0] = past point)
        while len(self.path) > 1:
            dx = self.path[0][0] - self.current_pose.position.x
            dy = self.path[0][1] - self.current_pose.position.y
            if math.hypot(dx, dy) < self.follow_distance_thresh:
                self.path.pop(0)
            else:
                break

        if len(self.path) < 2:
            self.publish_zero()
            return

        # Choose a lookahead waypoint: pick first path point at >= lookahead_dist or cap by max_lookahead_idx
        look_idx = 1
        accumulated = 0.0
        for i in range(1, min(len(self.path), 1 + self.max_lookahead_idx)):
            prev = self.path[i-1]
            cur = self.path[i]
            accumulated += math.dist(prev, cur)
            if accumulated >= self.lookahead_dist:
                look_idx = i
                break
        lookahead = self.path[look_idx]
        # compute attractive vector in global frame
        dx = lookahead[0] - self.current_pose.position.x
        dy = lookahead[1] - self.current_pose.position.y
        dist_to_look = math.hypot(dx, dy)

        # Transform attractive vector to robot frame
        yaw = self._quat_to_yaw(self.current_pose.orientation)
        att_x_r =  math.cos(-yaw)*dx - math.sin(-yaw)*dy
        att_y_r =  math.sin(-yaw)*dx + math.cos(-yaw)*dy

        # Immediate stop check: only if obstacle very close in front cone
        immediate_stop = False
        if self.latest_scan:
            immediate_stop = self._scan_front_blocking(self.latest_scan, self.stop_dist, self.immediate_stop_front_angle)
        if immediate_stop:
            # start escape maneuver if not already
            now = self.get_clock().now().seconds_nanoseconds()[0] + self.get_clock().now().seconds_nanoseconds()[1]*1e-9
            if not self.in_escape:
                self.in_escape = True
                self.escape_start_time = now
                self.escape_phase = 'reverse'
                self.get_logger().warn('Controller: immediate stop — starting escape maneuver')
            # perform escape maneuver (reverse then rotate)
            elapsed = now - self.escape_start_time
            if self.escape_phase == 'reverse' and elapsed < self.escape_reverse_time:
                cmd = Twist()
                cmd.linear.x = -0.08
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                return
            elif self.escape_phase == 'reverse':
                self.escape_phase = 'rotate'
                self.escape_start_time = now
            if self.escape_phase == 'rotate' and (now - self.escape_start_time) < self.escape_rotate_time:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.6
                self.cmd_pub.publish(cmd)
                return
            # finished escape -> request replan subject to cooldown
            self.in_escape = False
            self.escape_phase = None
            if (now - self.last_replan_time) >= self.replan_request_cooldown:
                self.get_logger().warn('Controller: escape finished — requesting replan')
                b = Bool(); b.data = True; self.replan_pub.publish(b)
                self.last_replan_time = now
            self.publish_zero()
            return

        # Compute repulsive vector from scan in robot frame
        rep_x = 0.0
        rep_y = 0.0
        closest_range = float('inf')
        if self.latest_scan:
            angle = self.latest_scan.angle_min
            for r in self.latest_scan.ranges:
                if r <= 0 or math.isinf(r):
                    angle += self.latest_scan.angle_increment
                    continue
                if r > self.avoidance_radius:
                    angle += self.latest_scan.angle_increment
                    continue
                # robot-frame coordinate of obstacle point
                ox = r * math.cos(angle)
                oy = r * math.sin(angle)
                # angle weighting: front points weigh more
                weight_angle = math.cos(angle)  # cos near 0 => 1 (front), near +-pi/2 => ~0
                # distance-based magnitude
                if r < self.avoidance_influence:
                    mag = (self.avoidance_influence - r) / (self.avoidance_influence + 1e-6)
                    influence = 1.0 * mag
                else:
                    mag = (self.avoidance_radius - r) / (self.avoidance_radius + 1e-6)
                    influence = 0.35 * mag
                # repulsion vector is away from obstacle (-ox,-oy)
                rep_x += (-ox) * influence * max(0.0, weight_angle)
                rep_y += (-oy) * influence * max(0.0, weight_angle)
                if r < closest_range:
                    closest_range = r
                angle += self.latest_scan.angle_increment

        # combine attractive and repulsive vectors (robot frame)
        # scale attraction with distance
        att_scale = 1.0
        if dist_to_look > 1.0:
            att_scale = 1.2
        combined_x = att_scale * att_x_r + self.avoidance_strength * rep_x
        combined_y = att_scale * att_y_r + self.avoidance_strength * rep_y

        # desired heading in robot frame
        desired_angle_robot = math.atan2(combined_y, combined_x)
        # angle diff (robot frame) is simply desired_angle_robot
        angle_diff = self._normalize(desired_angle_robot)

        # compute linear speed base on distance and obstacles
        speed_base = min(self.max_linear_speed, 0.6 * dist_to_look)
        if closest_range != float('inf'):
            speed_base = min(speed_base, max(0.05, self.max_linear_speed * (closest_range / (self.avoidance_influence + 1e-6))))
        # reduce speed if requested heading is large (sharp turn)
        if abs(angle_diff) > 0.7:
            speed_base *= 0.2
        elif abs(angle_diff) > 0.4:
            speed_base *= 0.45

        # tune if repulsion is opposing the attraction (dot product negative)
        dot = att_x_r * combined_x + att_y_r * combined_y
        if dot < 0:
            # strong repulsion causing backwards vector -> slow and focus on rotation
            speed_base *= 0.25

        # angular command proportional to angle_diff
        ang_cmd = max(-self.max_angular_speed, min(self.max_angular_speed, 1.3 * angle_diff))
        lin_cmd = max(0.0, min(self.max_linear_speed, speed_base))

        # publish twist
        cmd = Twist()
        cmd.linear.x = lin_cmd
        cmd.angular.z = ang_cmd
        self.cmd_pub.publish(cmd)

        # ---------- blocked detection (final goal progress) ----------
        final_goal = self.path[-1]
        goal_dist = math.hypot(final_goal[0] - self.current_pose.position.x,
                               final_goal[1] - self.current_pose.position.y)
        if self.prev_goal_dist is None:
            self.prev_goal_dist = goal_dist
            self.blocked_counter = 0
        else:
            if goal_dist > (self.prev_goal_dist - 0.02):
                self.blocked_counter += 1
            else:
                self.blocked_counter = 0
                self.prev_goal_dist = goal_dist

        if self.blocked_counter >= self.replan_threshold_cycles:
            now = self.get_clock().now().seconds_nanoseconds()[0] + self.get_clock().now().seconds_nanoseconds()[1]*1e-9
            if (now - self.last_replan_time) >= self.replan_request_cooldown:
                self.get_logger().warn('Controller: stuck -> requesting replan')
                b = Bool(); b.data = True; self.replan_pub.publish(b); self.last_replan_time = now
            self.blocked_counter = 0
            self.prev_goal_dist = None

    # ---------- Utility helpers ----------
    def _scan_front_blocking(self, scan: LaserScan, threshold: float, front_angle: float):
        """
        Return True if any scan point within 'threshold' exists inside +/- front_angle (radians).
        front_angle = e.g. 25 deg => only consider points near forward direction.
        """
        if scan is None:
            return False
        angle = scan.angle_min
        for r in scan.ranges:
            if r <= 0 or math.isinf(r):
                angle += scan.angle_increment
                continue
            if abs(angle) <= front_angle and r < threshold:
                return True
            angle += scan.angle_increment
        return False

    def publish_zero(self):
        t = Twist()
        self.cmd_pub.publish(t)

    def _quat_to_yaw(self, q):
        siny_cosp = 2.0*(q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize(self, a):
        # normalize angle to [-pi, pi]
        while a > math.pi:
            a -= 2.0*math.pi
        while a < -math.pi:
            a += 2.0*math.pi
        return a

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
