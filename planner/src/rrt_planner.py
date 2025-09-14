#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
import math, random, time

class PlannerNode(Node):
    def __init__(self):
        super().__init__('rrt_planner_node')

        # ---------- Parameters ----------
        self.declare_parameter('max_nodes', 3000)
        self.declare_parameter('step_size', 0.35)
        self.declare_parameter('goal_threshold', 0.05)
        self.declare_parameter('world_bounds', [-5.0, 5.0, -5.0, 5.0])
        self.declare_parameter('min_clearance', 0.15)
        self.declare_parameter('robot_radius', 0.15)
        self.declare_parameter('goal_sample_prob', 0.12)

        self.declare_parameter('replan_cooldown', 2.0)
        self.declare_parameter('planning_timeout', 1.5)

        # map usage
        self.declare_parameter('use_map_if_available', True)
        self.declare_parameter('map_inflation_cells', 1)  # inflate obstacles in grid

        # advanced
        self.declare_parameter('connect_goal_steps', 6)  # intermediate samples when trying to directly connect to goal
        self.declare_parameter('segment_check_step', 0.05)  # sampling resolution when checking segment collisions

        # load parameters
        self.max_nodes = self.get_parameter('max_nodes').value
        self.step_size = self.get_parameter('step_size').value
        self.goal_threshold = self.get_parameter('goal_threshold').value
        self.world_bounds = self.get_parameter('world_bounds').value
        self.min_clearance = self.get_parameter('min_clearance').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.goal_sample_prob = self.get_parameter('goal_sample_prob').value

        self.replan_cooldown = self.get_parameter('replan_cooldown').value
        self.planning_timeout = self.get_parameter('planning_timeout').value

        self.use_map_if_available = self.get_parameter('use_map_if_available').value
        self.map_inflation_cells = int(self.get_parameter('map_inflation_cells').value)

        self.connect_goal_steps = int(self.get_parameter('connect_goal_steps').value)
        self.segment_check_step = float(self.get_parameter('segment_check_step').value)

        # ---------- State ----------
        self.current_pose = None
        self.goal = None
        self.obstacles = []       # lidar points in robot frame
        self.obstacles_odom = []  # lidar points in odom frame
        self.is_planning = False
        self.last_replan_time = 0.0

        # occupancy grid (optional)
        self.map = None    # nav_msgs/OccupancyGrid
        self.map_data = None
        self.map_info = None

        # Publishers & Subscribers
        self.path_pub = self.create_publisher(Path, '/rrt_path', 10)
        self.status_pub = self.create_publisher(String, '/rrt_status', 10)

        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 20)
        self.create_subscription(Bool, '/replan_request', self.replan_cb, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 1)

        self.create_timer(0.5, self.timer_cb)

        self.get_logger().info('PlannerNode initialized (improved RRT with map support & partial path fallback)')

    # ---------- Callbacks ----------
    def goal_cb(self, msg: PoseStamped):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'New goal: {self.goal}')
        self.is_planning = True
        self.last_replan_time = 0.0  # allow immediate planning for new goal

    def odom_cb(self, msg: Odometry):
        self.current_pose = msg.pose.pose
        if self.obstacles:
            self._build_obstacles_odom()

    def scan_cb(self, msg: LaserScan):
        pts = []
        angle = msg.angle_min
        for r in msg.ranges:
            if r > msg.range_min and r < msg.range_max and not math.isinf(r):
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                pts.append((x, y))
            angle += msg.angle_increment
        self.obstacles = pts
        if self.current_pose:
            self._build_obstacles_odom()

    def map_cb(self, msg: OccupancyGrid):
        # store map for occupancy checks
        self.map = msg
        self.map_info = msg.info
        self.map_data = msg.data  # flattened list
        self.get_logger().debug('Map received: size %d x %d' % (self.map_info.width, self.map_info.height))

    def replan_cb(self, msg: Bool):
        now = self.get_clock().now().seconds_nanoseconds()[0] + self.get_clock().now().seconds_nanoseconds()[1]*1e-9
        if not msg.data:
            return
        if self.is_planning:
            self.get_logger().debug('Planner: already planning — ignoring replan request')
            return
        if now - self.last_replan_time < self.replan_cooldown:
            self.get_logger().debug('Planner: replan request in cooldown — ignoring')
            return
        self.get_logger().info('Replan requested')
        self.is_planning = True
        self.last_replan_time = now

    def timer_cb(self):
        if self.is_planning:
            self.plan_and_publish()
            self.is_planning = False

    # ---------- Planning ----------
    def plan_and_publish(self):
        if self.current_pose is None or self.goal is None:
            self.get_logger().warn('Planner: missing start or goal')
            self._publish_status('failed')
            return

        start = (self.current_pose.position.x, self.current_pose.position.y)
        goal = self.goal

        # quick checks: if start or goal inside known obstacle (map or lidar), warn/attempt slight shift
        if self._point_in_inflated_obstacle(start):
            self.get_logger().warn('Start appears inside inflated obstacle; planner will try but result may fail')
        if self._point_in_inflated_obstacle(goal):
            self.get_logger().warn('Goal appears inside inflated obstacle; planner will try but result may fail')

        self.get_logger().info('Starting RRT planning...')
        nodes = [start]
        parents = {start: None}
        candidate_paths = []

        start_time = time.time()
        best_node = start
        best_dist = math.dist(start, goal)

        for i in range(self.max_nodes):
            # time-based timeout
            if (time.time() - start_time) > self.planning_timeout:
                self.get_logger().warn('Planner: planning timeout reached, aborting search')
                break

            # sample (with goal bias)
            if random.random() < self.goal_sample_prob:
                rnd = goal
            else:
                rnd = self._sample_free()

            # nearest node
            nearest = min(nodes, key=lambda n: (n[0]-rnd[0])**2 + (n[1]-rnd[1])**2)

            # steer: step from nearest toward rnd
            theta = math.atan2(rnd[1]-nearest[1], rnd[0]-nearest[0])
            new_node = (nearest[0] + self.step_size*math.cos(theta),
                        nearest[1] + self.step_size*math.sin(theta))

            # collision check for node and segment
            if self._in_collision(new_node, clearance=self.min_clearance):
                continue
            if not self._segment_free(nearest, new_node, step=self.segment_check_step, clearance=self.min_clearance):
                continue

            # append
            nodes.append(new_node)
            parents[new_node] = nearest

            # update best node (closest encountered node to goal) for partial fallback
            d_to_goal = math.dist(new_node, goal)
            if d_to_goal < best_dist:
                best_dist = d_to_goal
                best_node = new_node

            # Attempt direct connect to goal from new_node (in multiple small steps)
            if self._try_connect_to_goal(new_node, goal):
                path = self._extract_path(new_node, parents)
                # append exact goal as final point
                path.append(goal)
                candidate_paths.append(path)
                # **we don't break immediately** — still allow finding other candidates, but you could break to speed up
                # break

        # If any candidate paths, choose best
        if candidate_paths:
            best = min(candidate_paths, key=self._path_cost)
            best_smoothed = self._smooth(best, iterations=60)
            self._publish_path(best_smoothed)
            self._publish_status('success')
            self.get_logger().info(f'Planner: SUCCESS — published path with {len(best_smoothed)} points')
            return

        # No full solution found — fallback: publish partial path to best_node found (if improved)
        if best_node != start:
            self.get_logger().info('Planner: no full solution; returning partial path toward best node')
            partial_path = self._extract_path(best_node, parents)
            # optionally append an intermediate point toward goal (one step) if free
            # but ensure that intermediate segment is collision-free
            try_next = (best_node[0] + (goal[0]-best_node[0]) * 0.5,
                        best_node[1] + (goal[1]-best_node[1]) * 0.5)
            if not self._in_collision(try_next, clearance=self.min_clearance) and \
               self._segment_free(best_node, try_next, step=self.segment_check_step, clearance=self.min_clearance):
                partial_path.append(try_next)
            # publish partial path
            self._publish_path(partial_path)
            self._publish_status('partial')
            self.get_logger().info(f'Planner: PARTIAL — published partial path with {len(partial_path)} points (dist to goal {best_dist:.3f})')
            return

        # Nothing found
        self.get_logger().warn('Planner: no solution within node limit or timeout, and no partial progress')
        self._publish_status('failed')

    # ---------- Utilities ----------
    def _sample_free(self):
        """Sample a point either uniformly in world bounds, or from map free cells if map available."""
        if self.use_map_if_available and self.map is not None and self.map_info is not None:
            # sample by picking random free cell
            # simple approach: random attempts until free cell found (bounded tries)
            for _ in range(50):
                idx_x = random.randint(0, self.map_info.width-1)
                idx_y = random.randint(0, self.map_info.height-1)
                i = idx_y * self.map_info.width + idx_x
                val = self.map_data[i]
                if val == 0:  # free cell
                    # convert map index to world coords
                    wx = self.map_info.origin.position.x + (idx_x + 0.5) * self.map_info.resolution
                    wy = self.map_info.origin.position.y + (idx_y + 0.5) * self.map_info.resolution
                    return (wx, wy)
            # fallback to uniform
        # uniform sample in world bounds
        return (random.uniform(self.world_bounds[0], self.world_bounds[1]),
                random.uniform(self.world_bounds[2], self.world_bounds[3]))

    def _point_in_inflated_obstacle(self, p):
        """Check if point p is inside inflated obstacle (map or lidar)."""
        if self.use_map_if_available and self.map is not None:
            return self._map_in_collision(p, inflation=self.map_inflation_cells)
        # else use lidar obstacles
        for (ox, oy) in getattr(self, 'obstacles_odom', []):
            if math.hypot(p[0]-ox, p[1]-oy) < (self.min_clearance + self.robot_radius):
                return True
        return False

    def _in_collision(self, point, clearance=None):
        """Collision check: prefer map lookup if map exists, otherwise use lidar points with inflated radius."""
        if clearance is None:
            clearance = self.min_clearance
        if self.use_map_if_available and self.map is not None:
            return self._map_in_collision(point, inflation=math.ceil((clearance + self.robot_radius) / self.map_info.resolution))
        # lidar-based
        eff = clearance + self.robot_radius
        if not getattr(self, 'obstacles_odom', None):
            return False
        x, y = point
        for (ox, oy) in self.obstacles_odom:
            if math.hypot(x-ox, y-oy) < eff:
                return True
        return False

    def _map_in_collision(self, point, inflation=0):
        """Check occupancy grid: returns True if cell or neighborhood flagged occupied."""
        if self.map is None:
            return False
        mx = int((point[0] - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((point[1] - self.map_info.origin.position.y) / self.map_info.resolution)
        w = self.map_info.width
        h = self.map_info.height
        for dx in range(-inflation, inflation+1):
            for dy in range(-inflation, inflation+1):
                ix = mx + dx
                iy = my + dy
                if ix < 0 or iy < 0 or ix >= w or iy >= h:
                    # treat out-of-map as occupied
                    return True
                val = self.map_data[iy*w + ix]
                if val > 50:  # occupied threshold
                    return True
        return False

    def _segment_free(self, a, b, step=0.05, clearance=None):
        pts = self._interpolate(a, b, step)
        for p in pts:
            if self._in_collision(p, clearance=clearance):
                return False
        return True

    def _try_connect_to_goal(self, node, goal):
        """Try to connect node -> goal by checking several small steps (in case goal isn't exactly within goal_threshold)."""
        # Fast accept if close
        if math.dist(node, goal) < self.goal_threshold:
            return True
        # create intermediate points
        for t in range(1, self.connect_goal_steps+1):
            alpha = t / float(self.connect_goal_steps)
            p = (node[0] + alpha*(goal[0]-node[0]), node[1] + alpha*(goal[1]-node[1]))
            if self._in_collision(p, clearance=self.min_clearance):
                return False
        return True

    def _build_obstacles_odom(self):
        odom_pts = []
        yaw = self._quat_to_yaw(self.current_pose.orientation)
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        for (cx, cy) in self.obstacles:
            ox = rx + (cx * math.cos(yaw) - cy * math.sin(yaw))
            oy = ry + (cx * math.sin(yaw) + cy * math.cos(yaw))
            odom_pts.append((ox, oy))
        self.obstacles_odom = odom_pts

    def _extract_path(self, node, parents):
        path = []
        cur = node
        while cur is not None:
            path.append(cur)
            cur = parents.get(cur, None)
        path.reverse()
        return path

    def _interpolate(self, p1, p2, step=0.05):
        x1, y1 = p1
        x2, y2 = p2
        dist = math.dist(p1, p2)
        if dist == 0:
            return [p1]
        steps = max(1, int(dist / step))
        pts = [(x1 + (x2-x1)*t/steps, y1 + (y2-y1)*t/steps) for t in range(steps+1)]
        return pts

    def _path_cost(self, path):
        length = sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1))
        penalty = 0.0
        obs_list = getattr(self, 'obstacles_odom', []) or []
        for i in range(len(path)-1):
            mid = ((path[i][0]+path[i+1][0])/2.0, (path[i][1]+path[i+1][1])/2.0)
            if self.use_map_if_available and self.map is not None:
                d = min([math.hypot(mid[0]-ox, mid[1]-oy) for (ox,oy) in obs_list], default=1e6)
            else:
                d = min((math.hypot(mid[0]-ox, mid[1]-oy) for (ox,oy) in obs_list), default=1e6)
            thr = self.min_clearance + self.robot_radius
            if d < thr:
                penalty += (thr - d)**2
        return length + penalty*10.0

    def _smooth(self, path, iterations=50):
        if len(path) < 3:
            return path[:]
        sm = path[:]
        for _ in range(iterations):
            if len(sm) < 3:
                break
            i = random.randint(0, len(sm)-2)
            j = random.randint(i+1, len(sm)-1)
            if j <= i+1:
                continue
            a, b = sm[i], sm[j]
            if self._segment_free(a, b, step=self.segment_check_step, clearance=self.min_clearance):
                sm = sm[:i+1] + sm[j:]
        return sm

    def _publish_path(self, path):
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        for (x,y) in path:
            p = PoseStamped()
            p.header.frame_id = 'odom'
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        self.path_pub.publish(msg)

    def _publish_status(self, s: str):
        st = String()
        st.data = s
        self.status_pub.publish(st)

    def _quat_to_yaw(self, q):
        siny_cosp = 2.0*(q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
