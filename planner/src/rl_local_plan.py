#!/usr/bin/env python3
"""
Improved Hybrid Local Planner skeleton for ROS2 (Python)

Key improvements vs. earlier skeleton:
- Topic names use valid tokens: /candidate_trajectory/cand_<i>
- Uses a single MultiThreadedExecutor instead of per-node threading
- RLDecisionNode now implements:
  - score smoothing (exponential moving average)
  - trajectory commitment / hysteresis (minimum commit duration + switch threshold)
  - simple continuity bonus (prefer candidates similar to committed trajectory)
  - basic safety rejection (min obstacle clearance)
- CandidateGenerator unchanged except topic names

This file is intended to run in ROS2 Humble+ with a TurtleBot waffle in Ignition.
Replace placeholders (policy loading, advanced scoring) as you iterate.

"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32

import numpy as np
import math
import time

# --------------------------- Utility functions ---------------------------

def wrap_angle(angle):
    """Wrap to [-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def forward_unicycle(x, y, theta, v, w, dt):
    # simple Euler integration of unicycle
    x_new = x + v * math.cos(theta) * dt
    y_new = y + v * math.sin(theta) * dt
    theta_new = wrap_angle(theta + w * dt)
    return x_new, y_new, theta_new


# --------------------------- Candidate Generator ---------------------------
class CandidateGenerator(Node):
    def __init__(self):
        super().__init__('candidate_generator')
        qos = QoSProfile(depth=10)

        # Parameters
        self.declare_parameter('num_candidates', 8)
        self.declare_parameter('horizon', 3.0)     # seconds
        self.declare_parameter('dt', 0.2)         # sec per step
        self.declare_parameter('max_speed', 0.5)  # m/s
        self.declare_parameter('max_omega', 1.0)  # rad/s

        self.N = self.get_parameter('num_candidates').get_parameter_value().integer_value
        self.horizon = self.get_parameter('horizon').get_parameter_value().double_value
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.max_omega = self.get_parameter('max_omega').get_parameter_value().double_value

        # Publishers: one topic per candidate with valid token names
        self.candidate_pubs = []
        for i in range(self.N):
            topic = f'/candidate_trajectory/cand_{i}'
            pub = self.create_publisher(Path, topic, qos)
            self.candidate_pubs.append(pub)

        # Subscribe to odom and global path to get local goal
        self.odom = None
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos)
        self.global_path = None
        self.create_subscription(Path, '/global_path', self.global_path_cb, qos)

        # Timer to publish candidates at regular rate
        self.timer = self.create_timer(0.2, self.timer_cb)  # 5 Hz
        self.get_logger().info('CandidateGenerator started')

    def odom_cb(self, msg: Odometry):
        self.odom = msg

    def global_path_cb(self, msg: Path):
        self.global_path = msg

    def compute_local_goal(self, lookahead=3.0):
        # If global_path is available, pick a pose at lookahead distance along it
        if not self.global_path or not self.global_path.poses:
            return None
        poses = self.global_path.poses
        if not self.odom:
            return None
        rx = self.odom.pose.pose.position.x
        ry = self.odom.pose.pose.position.y
        acc = 0.0
        prevx, prevy = rx, ry
        for p in poses:
            px = p.pose.position.x
            py = p.pose.position.y
            step = math.hypot(px - prevx, py - prevy)
            acc += step
            if acc >= lookahead:
                return p
            prevx, prevy = px, py
        return poses[-1]

    def timer_cb(self):
        if not self.odom:
            return
        local_goal = self.compute_local_goal(lookahead=3.0)
        if not local_goal:
            return

        # current robot pose
        rx = self.odom.pose.pose.position.x
        ry = self.odom.pose.pose.position.y
        # yaw extraction from quaternion (simple)
        q = self.odom.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        rtheta = math.atan2(siny, cosy)

        # Candidate generation: sample speeds & omegas
        speeds = np.linspace(self.max_speed * 0.2, self.max_speed, self.N)
        omegas = np.linspace(-self.max_omega, self.max_omega, self.N)

        for i in range(self.N):
            v = float(speeds[i % len(speeds)])
            w = float(omegas[(i * 2) % len(omegas)])
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'map'  # ensure frame matches your system

            x, y, th = rx, ry, rtheta
            t = 0.0
            while t < self.horizon:
                x, y, th = forward_unicycle(x, y, th, v, w, self.dt)
                ps = PoseStamped()
                ps.header = path_msg.header
                ps.pose.position.x = x
                ps.pose.position.y = y
                ps.pose.orientation.z = math.sin(th / 2.0)
                ps.pose.orientation.w = math.cos(th / 2.0)
                path_msg.poses.append(ps)
                t += self.dt

            # publish candidate
            self.candidate_pubs[i].publish(path_msg)


# --------------------------- RL Decision Node (with smoothing & commitment) ---------------------------
class RLDecisionNode(Node):
    def __init__(self):
        super().__init__('rl_decision_node')
        qos = QoSProfile(depth=10)

        # Parameters
        self.declare_parameter('num_candidates', 8)
        self.declare_parameter('scan_bins', 36)
        self.declare_parameter('policy_path', '')
        self.declare_parameter('score_alpha', 0.4)  # smoothing factor
        self.declare_parameter('switch_threshold', 1.10)  # new_score must be > cur_score*threshold to switch
        self.declare_parameter('min_commit_time', 0.8)  # seconds to commit before switching
        self.declare_parameter('continuity_lambda', 0.5)  # weight for similarity bonus
        self.declare_parameter('safety_distance', 0.25)  # min clearance (m)

        self.N = self.get_parameter('num_candidates').get_parameter_value().integer_value
        self.scan_bins = self.get_parameter('scan_bins').get_parameter_value().integer_value
        self.policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        self.alpha = self.get_parameter('score_alpha').get_parameter_value().double_value
        self.switch_threshold = self.get_parameter('switch_threshold').get_parameter_value().double_value
        self.min_commit_time = self.get_parameter('min_commit_time').get_parameter_value().double_value
        self.cont_lambda = self.get_parameter('continuity_lambda').get_parameter_value().double_value
        self.safety_distance = self.get_parameter('safety_distance').get_parameter_value().double_value

        # Subscribe to candidate topics
        self.candidate_paths = [None] * self.N
        for i in range(self.N):
            topic = f'/candidate_trajectory/cand_{i}'
            self.create_subscription(Path, topic, self._make_candidate_cb(i), qos)

        # Subscribe to sensors to build observation
        self.scan = None
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos)
        self.odom = None
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos)

        # Publishers
        self.selected_pub = self.create_publisher(Path, '/selected_trajectory', qos)
        self.selected_idx_pub = self.create_publisher(Int32, '/selected_idx', qos)

        # Scoring state
        self.smoothed_scores = np.zeros(self.N, dtype=np.float32)
        self.last_raw_scores = np.zeros(self.N, dtype=np.float32)

        # Commitment state
        self.committed_idx = None
        self.committed_ts = 0.0
        self.committed_path = None

        # Placeholder policy: None => random selection
        self.policy = None
        if self.policy_path:
            try:
                import torch
                self.get_logger().info(f'Loading policy from {self.policy_path}')
                self.policy = torch.load(self.policy_path)
                self.policy.eval()
            except Exception as e:
                self.get_logger().warn(f'Could not load policy: {e}. Using heuristic policy.')
                self.policy = None

        self.timer = self.create_timer(0.2, self.timer_cb)  # 5 Hz decision
        self.get_logger().info('RLDecisionNode started (with smoothing & commitment)')

    def _make_candidate_cb(self, idx):
        def cb(msg: Path):
            self.candidate_paths[idx] = msg
        return cb

    def scan_cb(self, msg: LaserScan):
        self.scan = msg

    def odom_cb(self, msg: Odometry):
        self.odom = msg

    def build_observation(self):
        # compact lidar: divide into bins and take min range per bin
        obs = []
        if self.scan is None:
            obs.extend([10.0] * self.scan_bins)
        else:
            ranges = np.array(self.scan.ranges)
            ranges = np.where(np.isnan(ranges), self.scan.range_max, ranges)
            L = len(ranges)
            bins = np.array_split(ranges, self.scan_bins)
            min_vals = [float(np.min(b)) for b in bins]
            obs.extend(min_vals)

        # odom: add vx, vy (approx), yaw
        if self.odom:
            vx = self.odom.twist.twist.linear.x
            vy = self.odom.twist.twist.linear.y
            q = self.odom.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)
            obs.extend([vx, vy, yaw])
        else:
            obs.extend([0.0, 0.0, 0.0])

        # Add local goal proxy: use endpoint of candidate 0 if available
        goal_dx = 0.0
        goal_dy = 0.0
        cp = next((c for c in self.candidate_paths if c), None)
        if cp and self.odom:
            last = cp.poses[-1].pose.position
            rx = self.odom.pose.pose.position.x
            ry = self.odom.pose.pose.position.y
            goal_dx = last.x - rx
            goal_dy = last.y - ry
        obs.extend([goal_dx, goal_dy])

        return np.array(obs, dtype=np.float32)

    def compute_candidate_score(self, idx):
        """Heuristic scoring combining progress, clearance, and similarity to committed path."""
        path = self.candidate_paths[idx]
        if path is None or not path.poses:
            return -1e6  # very bad

        # Progress: how close endpoint is to a proxy goal (we use endpoint distance as negative cost)
        end = path.poses[-1].pose.position
        if self.odom:
            rx = self.odom.pose.pose.position.x
            ry = self.odom.pose.pose.position.y
        else:
            rx = 0.0
            ry = 0.0
        dist_end = math.hypot(end.x - rx, end.y - ry)
        progress_score = -dist_end  # closer endpoint is better

        # Clearance: estimate min distance to obstacle from lidar bins
        clearance = 10.0
        if self.scan is not None:
            ranges = np.array(self.scan.ranges)
            ranges = np.where(np.isnan(ranges), self.scan.range_max, ranges)
            clearance = float(np.min(ranges))
        clearance_score = clearance  # larger clearance better

        # Continuity (similarity): if we have a committed path, reward candidates similar to it
        cont_score = 0.0
        if self.committed_path is not None and self.committed_path.poses:
            committed_end = self.committed_path.poses[-1].pose.position
            d = math.hypot(end.x - committed_end.x, end.y - committed_end.y)
            cont_score = -d  # smaller distance -> higher score

        # Combine (weights chosen heuristically; tune these)
        score = (1.0 * progress_score) + (0.7 * clearance_score) + (self.cont_lambda * cont_score)

        # Safety: if any point in path gets too close to obstacles (approximate using clearance), mark low
        if clearance < self.safety_distance:
            score -= 1000.0  # heavy penalty for unsafe candidate

        return score

    def pick_candidate_by_policy(self, obs):
        # If a learned policy is available, call it and get index.
        if self.policy is None:
            return None
        try:
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(obs).unsqueeze(0)
                out = self.policy(tensor)
                if isinstance(out, tuple):
                    out = out[0]
                idx = int(torch.argmax(out, dim=1).item())
                return idx
        except Exception as e:
            self.get_logger().warn(f'Policy call failed: {e}. Falling back to heuristic.')
            return None

    def decide_with_hysteresis(self):
        # Compute raw scores for each candidate
        raw_scores = np.array([self.compute_candidate_score(i) for i in range(self.N)], dtype=np.float32)
        self.last_raw_scores = raw_scores.copy()

        # Update smoothed scores (EMA)
        self.smoothed_scores = self.alpha * raw_scores + (1.0 - self.alpha) * self.smoothed_scores

        # Choose best by smoothed score
        best_idx = int(np.argmax(self.smoothed_scores))
        best_score = float(self.smoothed_scores[best_idx])

        # If nothing committed, commit immediately
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.committed_idx is None:
            self.committed_idx = best_idx
            self.committed_ts = now
            self.committed_path = self.candidate_paths[best_idx]
            return best_idx

        # Get current committed score (smoothed)
        cur_idx = self.committed_idx
        cur_score = float(self.smoothed_scores[cur_idx])

        # Safety check: if committed candidate vanished or became unsafe, allow immediate switch
        if self.candidate_paths[cur_idx] is None:
            self.get_logger().warn('Committed candidate missing; switching')
            self.committed_idx = best_idx
            self.committed_ts = now
            self.committed_path = self.candidate_paths[best_idx]
            return best_idx

        # If the best candidate is the same as committed -> keep following
        if best_idx == cur_idx:
            return cur_idx

        # else evaluate switching conditions
        time_since_commit = now - self.committed_ts
        # If min commit time not reached, only switch if new candidate is *much* better
        if time_since_commit < self.min_commit_time:
            if best_score > cur_score * (self.switch_threshold * 1.5):  # stricter while within commit time
                self.get_logger().info(f'Forced switch (early) from {cur_idx} to {best_idx}')
                self.committed_idx = best_idx
                self.committed_ts = now
                self.committed_path = self.candidate_paths[best_idx]
                return best_idx
            else:
                return cur_idx

        # If min commit time reached, allow switch if new_score sufficiently better
        if best_score > cur_score * self.switch_threshold:
            self.get_logger().info(f'Switching from {cur_idx} to {best_idx} (scores {cur_score:.2f} -> {best_score:.2f})')
            self.committed_idx = best_idx
            self.committed_ts = now
            self.committed_path = self.candidate_paths[best_idx]
            return best_idx

        # Otherwise stick with current
        return cur_idx

    def timer_cb(self):
        # decision loop
        # Build observation (not used with heuristic but kept for future policy)
        obs = self.build_observation()

        # If a policy exists, prefer policy decision (but still enforce safety/commitment)
        policy_idx = self.pick_candidate_by_policy(obs)
        if policy_idx is not None:
            # ensure policy candidate exists and is safe
            if self.candidate_paths[policy_idx] is not None:
                # accept policy choice but run through commitment mechanism
                # we can set smoothed_scores[policy_idx] artificially high to prefer it
                self.smoothed_scores[policy_idx] += 1.0

        # Decide using heuristic + hysteresis
        chosen_idx = self.decide_with_hysteresis()

        # Publish chosen path and index (if available)
        chosen_path = self.candidate_paths[chosen_idx]
        if chosen_path is None:
            # nothing to publish
            return

        # Safety final check: if path is unsafe (clearance < safety), override to stop
        if self.last_raw_scores[chosen_idx] < -900.0:
            # unsafe candidate
            self.get_logger().warn('Chosen candidate unsafe -> publishing empty stop path')
            stop_path = Path()
            stop_path.header.stamp = self.get_clock().now().to_msg()
            stop_path.header.frame_id = 'map'
            # create trivial 1-step path at current pose
            if self.odom:
                ps = PoseStamped()
                ps.header = stop_path.header
                ps.pose = self.odom.pose.pose
                stop_path.poses.append(ps)
            self.selected_pub.publish(stop_path)
            mi = Int32()
            mi.data = -1
            self.selected_idx_pub.publish(mi)
            return

        # Publish chosen
        self.selected_pub.publish(chosen_path)
        mi = Int32()
        mi.data = int(chosen_idx)
        self.selected_idx_pub.publish(mi)


# --------------------------- Main ---------------------------

def main(args=None):
    rclpy.init(args=args)

    cg = CandidateGenerator()
    rl = RLDecisionNode()

    executor = MultiThreadedExecutor()
    executor.add_node(cg)
    executor.add_node(rl)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        cg.destroy_node()
        rl.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
