#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Point, Pose
import random
import math
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
import time

class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0  # Cost from start to this node

class RRTPlanner(Node):
    def __init__(self):
        super().__init__('rrt_planner')
        
        # Parameters
        self.declare_parameter('max_iterations', 5000)
        self.declare_parameter('step_size', 0.2)  # meters
        self.declare_parameter('goal_sample_rate', 0.1)  # 10% chance to sample goal
        self.declare_parameter('goal_tolerance', 0.5)  # meters
        self.declare_parameter('obstacle_inflation', 0.3)  # Safety margin around obstacles (meters)
        self.declare_parameter('smoothing_iterations', 50)  # Path smoothing iterations
        
        self.max_iterations = self.get_parameter('max_iterations').value
        self.step_size = self.get_parameter('step_size').value
        self.goal_sample_rate = self.get_parameter('goal_sample_rate').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.obstacle_inflation = self.get_parameter('obstacle_inflation').value
        self.smoothing_iterations = self.get_parameter('smoothing_iterations').value
        
        # Map subscription
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10)
            
        # Goal subscription
        self.goal_sub = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.goal_callback,
            10)
            
        # Path publisher
        self.path_pub = self.create_publisher(
            Path,
            'global_path',
            10)
        
        # Odometry subscription
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/odom', 
            self.odom_callback, 
            10)
            
        # Initialize
        self.map = None
        self.map_info = None
        self.start_pose = None
        self.goal_pose = None
        self.node_list = []
        self.current_pose = None
        
        # Initialize inflated obstacle map
        self.inflated_map = None
        
        self.get_logger().info('RRT Global Planner initialized')
        
    def map_callback(self, msg):
        self.map = msg
        self.map_info = msg.info
        self.get_logger().info('Map received')
        
        # Create inflated obstacle map when we receive a map
        self.inflate_obstacles()
        
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        
    def goal_callback(self, msg):
        self.goal_pose = msg
        
        if self.current_pose is None:
            self.get_logger().warn('No odometry data available yet')
            return
            
        # Use current robot position as start pose
        self.start_pose = PoseStamped()
        self.start_pose.header = msg.header
        self.start_pose.pose = self.current_pose
        
        self.plan_path()
        
    def inflate_obstacles(self):
        """Inflate obstacles in the map for safety"""
        if self.map is None:
            return
            
        # Create a copy of the map data
        self.inflated_map = np.array(self.map.data).reshape((self.map_info.height, self.map_info.width))
        
        # Calculate inflation in grid cells
        inflation_cells = int(self.obstacle_inflation / self.map_info.resolution)
        
        # Find obstacle cells
        obstacle_indices = np.where(self.inflated_map > 50)
        
        # Temporary map for inflation
        inflated = np.copy(self.inflated_map)
        
        # Inflate obstacles
        for i in range(len(obstacle_indices[0])):
            y, x = obstacle_indices[0][i], obstacle_indices[1][i]
            
            # Inflate in a square around the obstacle
            y_min = max(0, y - inflation_cells)
            y_max = min(self.map_info.height - 1, y + inflation_cells)
            x_min = max(0, x - inflation_cells)
            x_max = min(self.map_info.width - 1, x + inflation_cells)
            
            for iy in range(y_min, y_max + 1):
                for ix in range(x_min, x_max + 1):
                    # Calculate distance to obstacle
                    distance = math.sqrt((ix - x)**2 + (iy - y)**2)
                    
                    # If within inflation radius, mark as obstacle
                    if distance <= inflation_cells:
                        inflated[iy, ix] = 100
        
        # Update inflated map
        self.inflated_map = inflated
        self.get_logger().info('Obstacles inflated')
        
    def plan_path(self):
        if self.map is None:
            self.get_logger().warn('No map available')
            return
            
        if self.start_pose is None or self.goal_pose is None:
            self.get_logger().warn('Start or goal pose not set')
            return
            
        start_time = time.time()
            
        # Initialize RRT
        self.node_list = []
        start_node = RRTNode(self.start_pose.pose.position.x, self.start_pose.pose.position.y)
        self.node_list.append(start_node)
        
        goal_reached = False
        
        # Run RRT algorithm
        for i in range(self.max_iterations):
            # Sample random point with bias towards goal
            if random.random() < self.goal_sample_rate:
                # Sample goal point
                rnd = RRTNode(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y)
            else:
                # Sample random point from free space
                rnd = self.sample_free_space()
                
            # Find nearest node
            nearest_ind = self.nearest_node_index(rnd)
            nearest_node = self.node_list[nearest_ind]
            
            # Steer towards random point
            new_node = self.steer(nearest_node, rnd)
            
            # Check if path is collision free
            if self.collision_free(nearest_node, new_node):
                # Calculate cost for the new node
                new_node.cost = nearest_node.cost + self.dist(nearest_node, new_node)
                
                self.node_list.append(new_node)
                
                # Check if we're near the goal
                dist_to_goal = self.dist(new_node, self.goal_pose.pose.position)
                if dist_to_goal <= self.goal_tolerance:
                    # Add final connection to goal
                    goal_node = RRTNode(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y)
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + dist_to_goal
                    self.node_list.append(goal_node)
                    self.get_logger().info(f'Path found after {i+1} iterations')
                    goal_reached = True
                    break
        
        if not goal_reached:
            self.get_logger().warn('Could not reach goal within iteration limit')
            
        end_time = time.time()
        self.get_logger().info(f'Planning time: {end_time - start_time:.3f} seconds')
        
        # Generate and smooth the path
        path = self.generate_path()
        
        # Smooth path
        if len(path.poses) > 2:
            path = self.smooth_path(path)
            
        # Publish path
        self.path_pub.publish(path)
        
    def sample_free_space(self):
        """Sample a random point from free space in the map"""
        if self.map is None:
            # Default if no map is available yet
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            return RRTNode(x, y)
            
        # Get map bounds in world coordinates
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        
        width_world = self.map_info.width * self.map_info.resolution
        height_world = self.map_info.height * self.map_info.resolution
        
        max_attempts = 100
        for _ in range(max_attempts):
            # Sample random point in world coordinates
            x = random.uniform(origin_x, origin_x + width_world)
            y = random.uniform(origin_y, origin_y + height_world)
            
            # Check if point is in free space
            if self.is_free(x, y):
                return RRTNode(x, y)
                
        # If we can't find a free cell after max attempts, just return a random point
        # This should be rare but prevents infinite loops
        self.get_logger().warn('Could not find free cell for sampling after max attempts')
        x = random.uniform(origin_x, origin_x + width_world)
        y = random.uniform(origin_y, origin_y + height_world)
        return RRTNode(x, y)
    
    def is_free(self, x, y):
        """Check if a point in world coordinates is in free space"""
        if self.inflated_map is None or self.map_info is None:
            return True
            
        # Convert world coordinates to grid indices
        grid_x = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        grid_y = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        
        # Check if indices are within map bounds
        if grid_x < 0 or grid_x >= self.map_info.width or grid_y < 0 or grid_y >= self.map_info.height:
            return False
            
        # Check if cell is free (using inflated map)
        return self.inflated_map[grid_y, grid_x] < 50
    
    def nearest_node_index(self, rnd_node):
        """Find the index of the nearest node to the given node"""
        dlist = [self.dist_squared(node, rnd_node) for node in self.node_list]
        return dlist.index(min(dlist))
    
    def steer(self, from_node, to_node):
        """Create a new node by steering from one node towards another with step size limit"""
        # Get direction
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dist = math.sqrt(dx**2 + dy**2)
        
        # Scale to step size
        if dist > self.step_size:
            dx = dx * self.step_size / dist
            dy = dy * self.step_size / dist
            
        new_node = RRTNode(from_node.x + dx, from_node.y + dy)
        new_node.parent = from_node
        return new_node
    
    def collision_free(self, from_node, to_node):
        """Check if the path between two nodes is collision free"""
        if self.inflated_map is None:
            return True
            
        # Interpolate points along the line
        dist = self.dist(from_node, to_node)
        if dist < self.map_info.resolution:
            return self.is_free(to_node.x, to_node.y)
            
        # Number of points to check along the line
        n_points = max(2, int(dist / (self.map_info.resolution * 0.5)))
        
        for i in range(n_points):
            # Interpolate point
            t = i / (n_points - 1)
            x = from_node.x + t * (to_node.x - from_node.x)
            y = from_node.y + t * (to_node.y - from_node.y)
            
            # Check if point is free
            if not self.is_free(x, y):
                return False
                
        return True
    
    def dist(self, node, point):
        """Calculate Euclidean distance between a node and a point"""
        if hasattr(point, 'x'):  # If point is a node or has x,y attributes
            return math.sqrt((node.x - point.x)**2 + (node.y - point.y)**2)
        else:  # If point is a geometry_msgs Point
            return math.sqrt((node.x - point.x)**2 + (node.y - point.y)**2)
    
    def dist_squared(self, node1, node2):
        """Calculate squared Euclidean distance between two nodes"""
        return (node1.x - node2.x)**2 + (node1.y - node2.y)**2
    
    def generate_path(self):
        """Generate a Path message from the RRT node list"""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        
        # Check if we have a path
        if len(self.node_list) < 2:
            self.get_logger().warn('No path found')
            return path
            
        # Start from the goal node (last in the list)
        current_node = self.node_list[-1]
        
        # Traverse parent nodes to build path
        path_nodes = []
        while current_node is not None:
            path_nodes.append(current_node)
            current_node = current_node.parent
            
        # Reverse to get start-to-goal order
        path_nodes.reverse()
        
        # Create PoseStamped messages
        for node in path_nodes:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = node.x
            pose.pose.position.y = node.y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        
        return path

    def smooth_path(self, path):
        """Apply path smoothing to reduce sharp turns"""
        if len(path.poses) <= 2:
            return path
            
        smooth_path = Path()
        smooth_path.header = path.header
        
        # Copy the original path
        poses = [pose for pose in path.poses]
        
        # Weight for smoothing
        weight_data = 0.5      # How much to maintain original path points
        weight_smooth = 0.3    # How much to smooth the path
        
        # Create a copy for smoothing iterations
        smooth_x = [pose.pose.position.x for pose in poses]
        smooth_y = [pose.pose.position.y for pose in poses]
        
        # Keep start and goal fixed
        for _ in range(self.smoothing_iterations):
            for i in range(1, len(poses) - 1):  # Skip start and goal
                # Calculate smoothing update
                smooth_x[i] += weight_data * (poses[i].pose.position.x - smooth_x[i])
                smooth_x[i] += weight_smooth * (smooth_x[i-1] + smooth_x[i+1] - 2.0 * smooth_x[i])
                
                smooth_y[i] += weight_data * (poses[i].pose.position.y - smooth_y[i])
                smooth_y[i] += weight_smooth * (smooth_y[i-1] + smooth_y[i+1] - 2.0 * smooth_y[i])
                
                # Verify smoothed point is collision-free
                if not self.is_free(smooth_x[i], smooth_y[i]):
                    # Revert to original if collision
                    smooth_x[i] = poses[i].pose.position.x
                    smooth_y[i] = poses[i].pose.position.y
        
        # Create new path with smoothed points
        for i in range(len(poses)):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = smooth_x[i]
            pose.pose.position.y = smooth_y[i]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            smooth_path.poses.append(pose)
            
        return smooth_path

def main(args=None):
    rclpy.init(args=args)
    rrt_planner = RRTPlanner()
    rclpy.spin(rrt_planner)
    rrt_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()