#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import math
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

class TEBLocalPlanner(Node):
    """
    Simplified TEB-like local planner:
    - subscribes to global_path and /scan
    - converts scan points to map frame using TF
    - extracts a local segment of the global path
    - relaxes the local path with smoothing + repulsive obstacle forces (elastic band)
    - publishes relaxed local_path and cmd_vel (pure pursuit style)
    """

    def __init__(self):
        super().__init__('teb_local_planner')

        # Parameters (tune these)
        self.declare_parameter('max_vel_x', 0.3)
        self.declare_parameter('max_vel_theta', 1.0)
        self.declare_parameter('look_ahead_dist', 1.0)
        self.declare_parameter('local_segment_length', 20)        # points along global path to consider
        self.declare_parameter('elastic_iters', 25)               # relaxation iterations
        self.declare_parameter('smoothing_weight', 0.1)          # smoothing scale (0..1)
        self.declare_parameter('obstacle_inflation', 0.25)        # meters - distance to start repulsion
        self.declare_parameter('obstacle_repulsive_gain', 2.0)   # gain for repulsive force
        self.declare_parameter('min_point_spacing', 0.01)        # numerical stability
        self.declare_parameter('lookahead_pure_pursuit', 0.6)   # for velocity command

        self.max_vel_x = self.get_parameter('max_vel_x').value
        self.max_vel_theta = self.get_parameter('max_vel_theta').value
        self.look_ahead_dist = self.get_parameter('look_ahead_dist').value
        self.local_segment_length = int(self.get_parameter('local_segment_length').value)
        self.elastic_iters = int(self.get_parameter('elastic_iters').value)
        self.smoothing_weight = float(self.get_parameter('smoothing_weight').value)
        self.obstacle_inflation = float(self.get_parameter('obstacle_inflation').value)
        self.obstacle_repulsive_gain = float(self.get_parameter('obstacle_repulsive_gain').value)
        self.min_point_spacing = float(self.get_parameter('min_point_spacing').value)
        self.lookahead_pure_pursuit = float(self.get_parameter('lookahead_pure_pursuit').value)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # state
        self.global_path = None
        self.obstacles_robot = []   # last laser points in robot frame (list of (x,y))
        self.obstacles_map = []     # same points transformed to map frame (list of (x,y))
        self.robot_pose_map = None  # (x,y,yaw) in map frame

        # pubs / subs
        self.global_path_sub = self.create_subscription(Path, 'global_path', self.global_path_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, 20)

        self.local_path_pub = self.create_publisher(Path, 'local_path', 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # timer
        self.create_timer(0.07, self.control_loop)  # ~14 Hz

        self.get_logger().info('TEBLocalPlanner (elastic-band) initialized')

    # ---------------- callbacks ----------------
    def global_path_cb(self, msg: Path):
        self.global_path = msg
        self.get_logger().debug(f'Global path received: {len(msg.poses)} poses')

    def scan_cb(self, msg: LaserScan):
        # store obstacle points in robot frame (x forward, y left)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        mask = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        xs = ranges[mask] * np.cos(angles[mask])
        ys = ranges[mask] * np.sin(angles[mask])
        self.obstacles_robot = list(zip(xs.tolist(), ys.tolist()))

    # ---------------- main loop ----------------
    def control_loop(self):
        # need path and TF map->base
        if self.global_path is None or len(self.global_path.poses) < 2:
            return

        # lookup robot pose in map frame
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except TransformException as ex:
            # try base_footprint if base_link missing
            try:
                t = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            except Exception as e:
                self.get_logger().warn(f'TF lookup failed: {ex}')
                return

        rx = t.transform.translation.x
        ry = t.transform.translation.y
        q = t.transform.rotation
        ryaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.robot_pose_map = (rx, ry, ryaw)

        # transform obstacles to map frame for obstacle constraints
        self._transform_obstacles_to_map(rx, ry, ryaw)

        # extract local segment
        local_pts = self._extract_local_segment()
        if not local_pts or len(local_pts) < 2:
            return

        # run elastic band relaxation on local_pts (in map coordinates)
        relaxed = self._elastic_relaxation(local_pts, self.obstacles_map)

        # publish relaxed local path
        self._publish_local_path(relaxed)

        # compute velocity command to follow relaxed path (pure pursuit-like)
        cmd = self._compute_velocity_from_path(relaxed)
        self.cmd_pub.publish(cmd)

    # ---------------- helpers ----------------
    def _transform_obstacles_to_map(self, rx, ry, ryaw):
        # rotate and translate robot-frame obstacles into map frame
        self.obstacles_map = []
        cos_y = math.cos(ryaw)
        sin_y = math.sin(ryaw)
        for (ox_r, oy_r) in self.obstacles_robot:
            oxm = rx + (ox_r * cos_y - oy_r * sin_y)
            oym = ry + (ox_r * sin_y + oy_r * cos_y)
            self.obstacles_map.append((oxm, oym))

    def _extract_local_segment(self):
        # find index of closest global pose to robot
        min_dist = float('inf')
        closest_idx = 0
        for i, p in enumerate(self.global_path.poses):
            px = p.pose.position.x
            py = p.pose.position.y
            d = math.hypot(px - self.robot_pose_map[0], py - self.robot_pose_map[1])
            if d < min_dist:
                min_dist = d
                closest_idx = i

        end_idx = min(len(self.global_path.poses), closest_idx + self.local_segment_length)
        pts = []
        for i in range(closest_idx, end_idx):
            p = self.global_path.poses[i].pose.position
            pts.append((p.x, p.y))
        return pts

    def _elastic_relaxation(self, pts, obstacles_map):
        """
        Simple elastic band relaxation with obstacle repulsion.
        pts: list of (x,y) in map frame (start..end)
        returns: new list of pts (relaxed)
        """
        if len(pts) < 2:
            return pts

        # convert to numpy array for vectorized ops
        P = np.array(pts, dtype=float)  # shape (N,2)
        N = P.shape[0]

        # fixed endpoints: keep P[0] (current robot vicinity) and P[-1] (goal of local segment) fixed
        for it in range(self.elastic_iters):
            # smoothing (internal force) - pull each interior point toward average of neighbors
            newP = P.copy()
            for i in range(1, N-1):
                # internal smoothing term
                prev = P[i-1]
                curr = P[i]
                nxt = P[i+1]
                smooth_term = (prev + nxt) / 2.0 - curr
                # apply smoothing
                newP[i] += self.smoothing_weight * smooth_term

            # obstacle repulsion (external force)
            if obstacles_map:
                # for each interior point, compute repulsive vector sum
                for i in range(1, N-1):
                    px, py = newP[i]
                    rep_x = 0.0
                    rep_y = 0.0
                    for (ox, oy) in obstacles_map:
                        dx = px - ox
                        dy = py - oy
                        d = math.hypot(dx, dy) + 1e-6
                        if d < self.obstacle_inflation:
                            # repulsive magnitude (quadratic or inverse)
                            # stronger when closer: use (inflation - d) / d
                            mag = (self.obstacle_inflation - d) / (d)
                            # scale and accumulate
                            rep_x += (dx / d) * mag
                            rep_y += (dy / d) * mag
                    # apply repulsive with gain
                    newP[i, 0] += self.obstacle_repulsive_gain * rep_x
                    newP[i, 1] += self.obstacle_repulsive_gain * rep_y

            # small step to avoid overshoot, enforce minimum spacing and keep endpoints fixed
            newP[0] = P[0]
            newP[-1] = P[-1]
            # enforce min spacing
            for i in range(1, N):
                d = math.hypot(newP[i,0] - newP[i-1,0], newP[i,1] - newP[i-1,1])
                if d < self.min_point_spacing:
                    # perturb slightly outward along neighbor direction
                    if d == 0:
                        newP[i,0] += 1e-3
                    else:
                        scale = (self.min_point_spacing / (d + 1e-6))
                        newP[i,0] = newP[i-1,0] + (newP[i,0] - newP[i-1,0]) * scale
                        newP[i,1] = newP[i-1,1] + (newP[i,1] - newP[i-1,1]) * scale

            P = newP

        # return as list of tuples
        return [(float(x), float(y)) for x,y in P]

    def _publish_local_path(self, pts):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for (x,y) in pts:
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.orientation.w = 1.0
            path.poses.append(p)
        self.local_path_pub.publish(path)

    def _compute_velocity_from_path(self, pts):
        """
        Pure-pursuit-like controller: choose lookahead point on pts at approx look_ahead_dist
        Compute heading error and output linear/angular commands.
        """
        cmd = Twist()
        if not pts or self.robot_pose_map is None:
            return cmd

        rx, ry, ryaw = self.robot_pose_map

        # find lookahead point
        acc = 0.0
        look_pt = None
        for i in range(1, len(pts)):
            seg = math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
            acc += seg
            if acc >= self.lookahead_pure_pursuit:
                look_pt = pts[i]
                break
        if look_pt is None:
            look_pt = pts[-1]

        dx = look_pt[0] - rx
        dy = look_pt[1] - ry
        # transform into robot frame
        rel_x = math.cos(-ryaw)*dx - math.sin(-ryaw)*dy
        rel_y = math.sin(-ryaw)*dx + math.cos(-ryaw)*dy

        target_yaw = math.atan2(rel_y, rel_x)
        # simple proportional controllers
        ang = 1.4 * target_yaw
        ang = max(-self.max_vel_theta, min(self.max_vel_theta, ang))

        # linear speed scaled by forward distance, reduced by angle
        dist = math.hypot(rel_x, rel_y)
        lin = self.max_vel_x * (1.0 - min(abs(target_yaw)/math.pi, 0.9))
        # slow down when close
        if dist < 0.2:
            lin = min(lin, 0.05)
        else:
            lin = min(lin, self.max_vel_x)

        cmd.linear.x = lin
        cmd.angular.z = ang
        return cmd

def main(args=None):
    rclpy.init(args=args)
    node = TEBLocalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()  