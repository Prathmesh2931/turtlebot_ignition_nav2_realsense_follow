#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan, Image
import random
import math
import tf2_ros
import cv_bridge
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Quaternion

class RRTPlanner(Node):
    def __init__(self):
        super().__init__("rrt_planner")

        # Parameters (tweak as required)
        self.declare_parameter('max_nodes', 5000)
        self.declare_parameter('step_size', 0.35)
        self.declare_parameter('goal_threshold', 0.15)
        self.declare_parameter('world_bounds', [-5.0, 5.0, -5.0, 5.0])
        self.declare_parameter('min_clearance', 0.20)   # minimum clearance (meters)
        self.declare_parameter('robot_radius', 0.15)    # robot radius to inflate obstacles
        self.declare_parameter('solution_sample_prob', 0.12) # prob to sample goal
        self.declare_parameter('follow_distance_thresh', 0.18)
        self.declare_parameter('scan_stop_dist', 0.15)  # emergency stop distance

        self.max_nodes = self.get_parameter('max_nodes').value
        self.step_size = self.get_parameter('step_size').value
        self.goal_threshold = self.get_parameter('goal_threshold').value
        self.world_bounds = self.get_parameter('world_bounds').value
        self.min_clearance = self.get_parameter('min_clearance').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.solution_sample_prob = self.get_parameter('solution_sample_prob').value
        self.follow_distance_thresh = self.get_parameter('follow_distance_thresh').value
        self.scan_stop_dist = self.get_parameter('scan_stop_dist').value

        # State
        self.start = None
        self.goal = None
        self.current_pose = None
        self.obstacles = []       # raw obstacles in robot frame: list of ((cx,cy),(w,h))
        self.obstacles_odom = []  # obstacles converted to odom frame: list of ((x,y),(w,h))
        self.path = []
        self.is_planning = False
        self.has_goal = False
        self.latest_scan = None

        # TF and CV bridge (kept for future use)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bridge = cv_bridge.CvBridge()

        # Publishers
        self.path_pub = self.create_publisher(Path, "/rrt_path", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/rrt_markers", 10)

        # Subscribers
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 20)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 20)
        self.depth_sub = self.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 1)

        # Control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("RRT Planner Node initialized")

    # ---------- Callbacks ----------
    def goal_callback(self, msg: PoseStamped):
        self.get_logger().info(f"Received new goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.has_goal = True
        self.is_planning = True

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose
        # update start for planning when planning triggered
        if self.is_planning and self.current_pose is not None:
            self.start = (self.current_pose.position.x, self.current_pose.position.y)
            # If we have raw laser obstacles, convert them to odom now (keep up-to-date)
            if self.obstacles:
                self._build_obstacles_odom()

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        # convert ranges to obstacle points in robot frame
        obstacles = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and not math.isinf(r):
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                # store small box around the point (w,h)
                obstacles.append(((x, y), (0.12, 0.12)))
            angle += msg.angle_increment
        self.obstacles = obstacles
        # build odom-frame obstacle list (if we have odom)
        if self.current_pose is not None:
            self._build_obstacles_odom()
        else:
            self.obstacles_odom = []

    def depth_callback(self, msg: Image):
        # placeholder for later depth-to-obstacles; not used now
        pass

    def _build_obstacles_odom(self):
        """Convert robot-frame obstacles to odom-frame obstacle centers (cached)."""
        odom_list = []
        for (cx, cy), (w, h) in self.obstacles:
            ox, oy = self._robot_frame_point_to_odom(cx, cy)
            odom_list.append(((ox, oy), (w, h)))
        self.obstacles_odom = odom_list

    # ---------- Main loop ----------
    def control_loop(self):
        if not self.has_goal or self.current_pose is None:
            return

        # Emergency stop if scan reports very close obstacle
        if (self.latest_scan is not None and
            any([(r < self.scan_stop_dist) for r in self.latest_scan.ranges if r > 0 and not math.isinf(r)])):
            self.get_logger().warn("Emergency stop: obstacle too close!")
            self.publish_zero_velocity()
            return

        if self.is_planning:
            self.plan_path()
            self.is_planning = False

        if self.path:
            self.follow_path()

    # ---------- Planner ----------
    def plan_path(self):
        if self.start is None or self.goal is None:
            self.get_logger().warn("Cannot plan: start or goal not set")
            return

        self.get_logger().info("Starting RRT planning (multi-solution search)...")

        nodes = [self.start]
        parents = {self.start: None}
        candidate_paths = []

        for i in range(self.max_nodes):
            # biased sampling toward goal
            if random.random() < self.solution_sample_prob:
                rnd = self.goal
            else:
                rnd = (random.uniform(self.world_bounds[0], self.world_bounds[1]),
                       random.uniform(self.world_bounds[2], self.world_bounds[3]))

            # nearest node
            nearest = min(nodes, key=lambda n: (n[0]-rnd[0])**2 + (n[1]-rnd[1])**2)

            # step toward rnd
            theta = math.atan2(rnd[1]-nearest[1], rnd[0]-nearest[0])
            new_node = (nearest[0] + self.step_size*math.cos(theta),
                        nearest[1] + self.step_size*math.sin(theta))

            # collision check for the node (inflated by robot radius + clearance)
            if self.in_collision(new_node, clearance=self.min_clearance):
                continue

            # check the segment from nearest -> new_node is also collision-free
            if not self.collision_free_segment(nearest, new_node, step=0.05, clearance=self.min_clearance):
                continue

            nodes.append(new_node)
            parents[new_node] = nearest

            # if close to goal, save candidate path (but continue searching)
            if math.dist(new_node, self.goal) < self.goal_threshold:
                path = self.extract_path(new_node, parents)
                candidate_paths.append(path)

        if not candidate_paths:
            self.get_logger().warn("No solutions found within node limit.")
            self.publish_markers(nodes, [])
            return

        # Evaluate candidate paths using cost function
        best_path = min(candidate_paths, key=self.path_cost)
        self.get_logger().info(f"Found {len(candidate_paths)} candidate paths. Best cost: {self.path_cost(best_path):.3f}")

        # Smooth the best path (respect clearance during smoothing)
        smoothed = self.smooth_path(best_path, iterations=80)
        self.path = smoothed
        self.publish_path(self.path)
        self.publish_markers(nodes, self.path)
        self.get_logger().info(f"Published best smoothed path with {len(self.path)} points.")

    # ---------- Collision & Cost ----------
    def in_collision(self, point, clearance=None):
        """
        Check if a point (in odom frame) violates clearance from obstacles.
        Uses cached self.obstacles_odom. The clearance used is (clearance + robot_radius).
        """
        if self.current_pose is None:
            return True  # unsafe to assume free before having pose

        if clearance is None:
            clearance = self.min_clearance

        effective_clear = clearance + self.robot_radius

        x, y = point
        if not self.obstacles_odom:
            return False

        for (ox, oy), (w, h) in self.obstacles_odom:
            # distance from obstacle center
            dist = math.hypot(x - ox, y - oy)
            # approximate obstacle "radius" from w,h (diagonal / 2)
            obs_radius = math.hypot(w/2.0, h/2.0)
            if dist < (effective_clear + obs_radius):
                return True
        return False

    def collision_free_segment(self, a, b, step=0.05, clearance=None):
        """Sample along segment a->b and ensure no sample point is in collision."""
        for p in self.interpolate(a, b, step=step):
            if self.in_collision(p, clearance=clearance):
                return False
        return True

    def path_cost(self, path):
        """
        Cost = path length + clearance penalty (heavy penalty for segments that come closer than threshold).
        Lower cost is better.
        """
        if not path or len(path) < 2:
            return float('inf')

        length = 0.0
        clearance_penalty = 0.0
        for i in range(len(path)-1):
            a = path[i]
            b = path[i+1]
            seg_len = math.dist(a, b)
            length += seg_len

            # Evaluate clearance at segment mid-point
            mid = ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0)
            min_dist = self._min_dist_to_obstacles(mid)
            # threshold for penalty
            threshold = self.min_clearance * 1.5 + self.robot_radius
            if min_dist < threshold:
                clearance_penalty += (threshold - min_dist) ** 2 * 15.0  # squared penalty

        return length + clearance_penalty

    def _min_dist_to_obstacles(self, point):
        """Return minimum euclidean distance from 'point' to any known obstacle center (odom frame)."""
        if not self.obstacles_odom:
            return float('inf')
        x, y = point
        min_d = float('inf')
        for (ox, oy), _ in self.obstacles_odom:
            d = math.hypot(x-ox, y-oy)
            if d < min_d:
                min_d = d
        return min_d

    def _robot_frame_point_to_odom(self, rx, ry):
        """Convert point from robot frame to odom frame using current_pose yaw (2D)."""
        if self.current_pose is None:
            return rx, ry
        yaw = self._quaternion_to_yaw(self.current_pose.orientation)
        odom_x = self.current_pose.position.x + (rx * math.cos(yaw) - ry * math.sin(yaw))
        odom_y = self.current_pose.position.y + (rx * math.sin(yaw) + ry * math.cos(yaw))
        return odom_x, odom_y

    # ---------- Path utilities ----------
    def extract_path(self, node, parents):
        path = []
        cur = node
        while cur is not None:
            path.append(cur)
            cur = parents.get(cur, None)
        path.reverse()
        return path

    def interpolate(self, p1, p2, step=0.05):
        x1, y1 = p1
        x2, y2 = p2
        dist = math.dist(p1, p2)
        if dist == 0:
            return [p1]
        steps = max(1, int(dist / step))
        pts = [(x1 + (x2-x1)*t/steps, y1 + (y2-y1)*t/steps) for t in range(steps+1)]
        return pts

    def smooth_path(self, path, iterations=50):
        """
        Random shortcut smoothing: pick two points and try to connect them directly if the straight line is collision-free.
        """
        if len(path) < 3:
            return path[:]
        smoothed = path[:]
        for _ in range(iterations):
            if len(smoothed) < 3:
                break
            i = random.randint(0, len(smoothed)-2)
            j = random.randint(i+1, len(smoothed)-1)
            if j <= i+1:
                continue
            p_i = smoothed[i]
            p_j = smoothed[j]
            # check collision along straight segment
            if self.collision_free_segment(p_i, p_j, step=0.05, clearance=self.min_clearance):
                # remove intermediate points between i and j
                smoothed = smoothed[:i+1] + smoothed[j:]
        return smoothed

    # ---------- Execution ----------
    def follow_path(self):
        if not self.path or len(self.path) < 2 or self.current_pose is None:
            self.publish_zero_velocity()
            return

        # remove reached points
        while len(self.path) > 1:
            dx0 = self.path[0][0] - self.current_pose.position.x
            dy0 = self.path[0][1] - self.current_pose.position.y
            if math.hypot(dx0, dy0) < self.follow_distance_thresh:
                self.path.pop(0)
            else:
                break

        if len(self.path) < 2:
            self.publish_zero_velocity()
            return

        target = self.path[1]
        dx = target[0] - self.current_pose.position.x
        dy = target[1] - self.current_pose.position.y
        distance = math.hypot(dx, dy)

        # If close enough to final goal, pop and stop when empty
        if distance < self.follow_distance_thresh and len(self.path) <= 2:
            self.get_logger().info("Goal reached (by follower).")
            self.path = []
            self.publish_zero_velocity()
            return

        current_yaw = self._quaternion_to_yaw(self.current_pose.orientation)
        target_angle = math.atan2(dy, dx)
        angle_diff = self._normalize_angle(target_angle - current_yaw)

        cmd = Twist()
        # angular control first if angle large
        if abs(angle_diff) > 0.35:
            cmd.linear.x = 0.0
            cmd.angular.z = max(-1.0, min(1.0, 1.2 * angle_diff))
        else:
            # forward motion with proportional linear and angular
            cmd.linear.x = min(0.35, 0.6 * distance)
            cmd.angular.z = max(-1.0, min(1.0, 0.9 * angle_diff))

        self.cmd_vel_pub.publish(cmd)

    def publish_zero_velocity(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    # ---------- Visualization ----------
    def publish_path(self, path_points):
        msg = Path()
        msg.header.frame_id = "odom"
        msg.header.stamp = self.get_clock().now().to_msg()
        for (x, y) in path_points:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def publish_markers(self, nodes, path):
        marker_array = MarkerArray()

        # nodes (points)
        node_marker = Marker()
        node_marker.header.frame_id = "odom"
        node_marker.header.stamp = self.get_clock().now().to_msg()
        node_marker.ns = "rrt_nodes"
        node_marker.id = 0
        node_marker.type = Marker.POINTS
        node_marker.action = Marker.ADD
        node_marker.pose.orientation.w = 1.0
        node_marker.scale.x = 0.05
        node_marker.scale.y = 0.05
        node_marker.color.a = 0.6
        node_marker.color.r = 0.0
        node_marker.color.g = 0.7
        node_marker.color.b = 0.0

        for (x, y) in nodes:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            node_marker.points.append(p)
        marker_array.markers.append(node_marker)

        # path
        if path:
            path_marker = Marker()
            path_marker.header.frame_id = "odom"
            path_marker.header.stamp = self.get_clock().now().to_msg()
            path_marker.ns = "rrt_path"
            path_marker.id = 1
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.pose.orientation.w = 1.0
            path_marker.scale.x = 0.08
            path_marker.color.a = 1.0
            path_marker.color.r = 1.0
            path_marker.color.g = 0.0
            path_marker.color.b = 0.0
            for (x, y) in path:
                p = Point()
                p.x = x
                p.y = y
                p.z = 0.05
                path_marker.points.append(p)
            marker_array.markers.append(path_marker)

        # obstacles visualization (odom frame)
        if self.obstacles_odom:
            idx = 2
            for (ox, oy), (w, h) in self.obstacles_odom:
                m = Marker()
                m.header.frame_id = "odom"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "rrt_obstacles"
                m.id = idx
                idx += 1
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = ox
                m.pose.position.y = oy
                m.pose.position.z = 0.0
                m.pose.orientation.w = 1.0
                m.scale.x = max(w, self.robot_radius*2.0)
                m.scale.y = max(h, self.robot_radius*2.0)
                m.scale.z = 0.05
                m.color.a = 0.6
                m.color.r = 0.9
                m.color.g = 0.2
                m.color.b = 0.2
                marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

    # ---------- helpers ----------
    def _quaternion_to_yaw(self, q: Quaternion):
        # yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

def main(args=None):
    rclpy.init(args=args)
    node = RRTPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
