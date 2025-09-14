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
    Improved TEB-like local planner with:
    - Path history for temporal smoothing
    - Better obstacle avoidance with progressive force scaling
    - Damping to reduce oscillations
    - Adaptive parameters based on obstacle density
    """

    def __init__(self):
        super().__init__('teb_local_planner')

        # Parameters (tune these)
        self.declare_parameter('max_vel_x', 0.3)
        self.declare_parameter('max_vel_theta', 1.0)
        self.declare_parameter('look_ahead_dist', 1.0)
        self.declare_parameter('local_segment_length', 20)        # points along global path to consider
        self.declare_parameter('elastic_iters', 30)               # relaxation iterations (increased)
        self.declare_parameter('smoothing_weight', 0.1)           # smoothing scale (0..1)
        self.declare_parameter('obstacle_inflation', 0.2)         # meters - distance to start repulsion (increased)
        self.declare_parameter('obstacle_repulsive_gain', 1.5)    # gain for repulsive force
        self.declare_parameter('min_point_spacing', 0.01)         # numerical stability
        self.declare_parameter('lookahead_pure_pursuit', 0.6)     # for velocity command
        self.declare_parameter('temporal_smoothing', 0.7)         # weight for previous path (0..1)
        self.declare_parameter('robot_radius', 0.14)               # robot footprint approx
        self.declare_parameter('damping_factor', 0.2)             # damping to reduce oscillations

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
        self.temporal_smoothing = float(self.get_parameter('temporal_smoothing').value)
        self.robot_radius = float(self.get_parameter('robot_radius').value)
        self.damping_factor = float(self.get_parameter('damping_factor').value)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # state
        self.global_path = None
        self.obstacles_robot = []   # last laser points in robot frame (list of (x,y))
        self.obstacles_map = []     # same points transformed to map frame (list of (x,y))
        self.robot_pose_map = None  # (x,y,yaw) in map frame
        self.prev_relaxed_path = None  # store previous relaxed path for temporal smoothing
        self.prev_cmd_vel = Twist()    # previous velocity command for smoothing

        # pubs / subs
        self.global_path_sub = self.create_subscription(Path, 'global_path', self.global_path_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, 20)

        self.local_path_pub = self.create_publisher(Path, 'local_path', 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # timer
        self.create_timer(0.07, self.control_loop)  # ~14 Hz

        self.get_logger().info('Improved TEBLocalPlanner (elastic-band) initialized')

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

        # Apply temporal smoothing with previous path if available
        if self.prev_relaxed_path is not None:
            relaxed = self._apply_temporal_smoothing(relaxed, self.prev_relaxed_path)
        
        # Store current path for next iteration
        self.prev_relaxed_path = relaxed

        # publish relaxed local path
        self._publish_local_path(relaxed)

        # compute velocity command to follow relaxed path (pure pursuit-like)
        cmd = self._compute_velocity_from_path(relaxed)
        
        # Smooth velocity commands to reduce jerkiness
        cmd = self._smooth_velocity_commands(cmd)
        
        self.cmd_pub.publish(cmd)
        self.prev_cmd_vel = cmd

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
        Improved elastic band relaxation with:
        - Progressive obstacle repulsion
        - Damping to reduce oscillations
        - Adaptive parameters based on obstacle density
        """
        if len(pts) < 2:
            return pts

        # convert to numpy array for vectorized ops
        P = np.array(pts, dtype=float)  # shape (N,2)
        N = P.shape[0]
        
        # Determine obstacle density for adaptive parameters
        obstacle_density = min(1.0, len(obstacles_map) / 100.0)  # normalized 0..1
        
        # Adjust parameters based on obstacle density
        adaptive_repulsive_gain = self.obstacle_repulsive_gain * (1.0 + obstacle_density)
        adaptive_smoothing = max(0.05, self.smoothing_weight * (1.0 - 0.5 * obstacle_density))

        # Keep track of forces for damping
        last_forces = np.zeros_like(P)

        # fixed endpoints: keep P[0] (current robot vicinity) and P[-1] (goal of local segment) fixed
        for it in range(self.elastic_iters):
            # Start with a fresh copy each iteration
            newP = P.copy()
            forces = np.zeros_like(P)
            
            # Apply smoothing forces (internal forces)
            for i in range(1, N-1):
                # internal smoothing term
                prev = P[i-1]
                curr = P[i]
                nxt = P[i+1]
                smooth_term = (prev + nxt) / 2.0 - curr
                
                # Store the smoothing force
                forces[i] += adaptive_smoothing * smooth_term

            # Apply obstacle repulsion forces (external forces)
            if obstacles_map:
                # for each interior point, compute repulsive vector sum
                for i in range(1, N-1):
                    px, py = P[i]
                    rep_x = 0.0
                    rep_y = 0.0
                    
                    for (ox, oy) in obstacles_map:
                        dx = px - ox
                        dy = py - oy
                        d = math.hypot(dx, dy) + 1e-6
                        
                        # Check if obstacle is within inflated radius + robot radius
                        inflation_dist = self.obstacle_inflation + self.robot_radius
                        
                        if d < inflation_dist:
                            # Progressive repulsion - stronger when closer
                            # Use cubic scaling for more aggressive close-range repulsion
                            rep_factor = (inflation_dist - d)**2 / (d + 0.01)
                            
                            # Apply minimum force for very close obstacles
                            rep_factor = max(rep_factor, 0.1)
                            
                            # Normalize direction vector
                            norm = math.sqrt(dx*dx + dy*dy)
                            if norm > 1e-6:
                                dir_x, dir_y = dx/norm, dy/norm
                                rep_x += dir_x * rep_factor
                                rep_y += dir_y * rep_factor
                    
                    # Store the repulsive force
                    forces[i, 0] += adaptive_repulsive_gain * rep_x
                    forces[i, 1] += adaptive_repulsive_gain * rep_y

            # Apply damping - mix current forces with previous forces
            if it > 0:
                forces = (1.0 - self.damping_factor) * forces + self.damping_factor * last_forces
            
            # Store forces for next iteration
            last_forces = forces.copy()
            
            # Apply forces to get new positions
            newP += forces
            
            # Keep endpoints fixed
            newP[0] = P[0]
            newP[-1] = P[-1]
            
            # enforce min spacing between points
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
    
    def _apply_temporal_smoothing(self, current_path, prev_path):
        """
        Apply temporal smoothing between current and previous paths
        to reduce path fluctuations over time.
        """
        # If path lengths don't match, resample the shorter one
        if len(current_path) != len(prev_path):
            # For simplicity, we'll just use the current path in this case
            return current_path
        
        # Create numpy arrays for vectorized operations
        curr = np.array(current_path)
        prev = np.array(prev_path)
        
        # Apply weighted average
        smoothed = (1.0 - self.temporal_smoothing) * curr + self.temporal_smoothing * prev
        
        # Convert back to list of tuples
        return [(float(x), float(y)) for x,y in smoothed]

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

        # Check for obstacles directly in front of robot for emergency braking
        emergency_stop = False
        for (ox_r, oy_r) in self.obstacles_robot:
            # Consider only obstacles in front of the robot
            if ox_r > 0 and abs(oy_r) < self.robot_radius:
                # If obstacle is too close, stop
                if ox_r < self.robot_radius + 0.15:  # 15cm safety margin
                    emergency_stop = True
                    self.get_logger().warn('Emergency stop: obstacle too close!')
                    break

        if emergency_stop:
            return cmd  # return zero velocity

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
        
        # Simple proportional controller for angular velocity
        ang = 1.4 * target_yaw
        ang = max(-self.max_vel_theta, min(self.max_vel_theta, ang))

        # Linear speed scaled by forward distance, reduced by angle
        dist = math.hypot(rel_x, rel_y)
        lin = self.max_vel_x * (1.0 - min(abs(target_yaw)/math.pi, 0.9))
        
        # Slow down when close to goal
        if dist < 0.2:
            lin = min(lin, 0.05)
        else:
            lin = min(lin, self.max_vel_x)

        # Scale down velocity based on obstacle density in front
        front_obstacles = [obs for obs in self.obstacles_robot 
                          if obs[0] > 0 and abs(obs[1]) < self.robot_radius + 0.2
                          and obs[0] < 1.0]
        
        if front_obstacles:
            # Find the closest front obstacle
            closest_dist = min([obs[0] for obs in front_obstacles])
            # Scale velocity based on distance (closer = slower)
            slowdown_factor = min(1.0, (closest_dist - self.robot_radius) / 0.5)
            slowdown_factor = max(0.2, slowdown_factor)  # don't go below 20% speed
            lin *= slowdown_factor

        cmd.linear.x = lin
        cmd.angular.z = ang
        return cmd
    
    def _smooth_velocity_commands(self, cmd):
        """
        Apply smoothing to velocity commands to reduce jerky motion
        """
        # Simple exponential smoothing with the previous velocity command
        smoothing_factor = 0.3  # How much to smooth (0-1)
        
        smooth_cmd = Twist()
        smooth_cmd.linear.x = (1.0 - smoothing_factor) * cmd.linear.x + smoothing_factor * self.prev_cmd_vel.linear.x
        smooth_cmd.angular.z = (1.0 - smoothing_factor) * cmd.angular.z + smoothing_factor * self.prev_cmd_vel.angular.z
        
        return smooth_cmd

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