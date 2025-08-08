#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.qos import QoSProfile, DurabilityPolicy
import numpy as np
import matplotlib.pyplot as plt
import heapq
import math
import os
# --- A* Algorithm Implementation (Unchanged) ---
class AStarNode:
    def __init__(self, parent=None, position=None):
        self.parent, self.position, self.g, self.h, self.f = parent, position, 0, 0, 0
    def __eq__(self, other): return self.position == other.position
    def __lt__(self, other): return self.f < other.f
    def __hash__(self): return hash(self.position)

def a_star_search(grid, start, end):
    start_node, end_node = AStarNode(None, start), AStarNode(None, end)
    open_list, closed_set = [], set()
    heapq.heappush(open_list, start_node)
    
    explored_path = []

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)
        explored_path.append(current_node.position)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1], explored_path

        for new_position in [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            
            if not (0 <= node_position[0] < len(grid) and 0 <= node_position[1] < len(grid[0])) \
               or grid[node_position[0]][node_position[1]] > 50 or node_position in closed_set:
                continue

            new_node = AStarNode(current_node, node_position)
            new_node.g = current_node.g + math.sqrt(new_position[0]**2 + new_position[1]**2)
            new_node.h = math.sqrt((node_position[0] - end_node.position[0])**2 + (node_position[1] - end_node.position[1])**2)
            new_node.f = new_node.g + new_node.h
            
            if any(open_node for open_node in open_list if new_node == open_node and new_node.g >= open_node.g):
                continue
                
            heapq.heappush(open_list, new_node)
            
    return None, explored_path

# --- Combined Planner and Controller Node ---
class AStarPlannerAndControllerNode(Node):
    def __init__(self):
        super().__init__('a_star_planner_and_controller_node')
        self.get_logger().info("A* Planner & Controller Node has started.")

        # --- Parameters ---
        # REMOVED: The 'start_point' is now dynamic.
        self.declare_parameter('goal_point', [-6.0, -9.0])
        self.declare_parameter('lookahead_distance', 0.8) 
        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('max_linear_speed', 0.6)
        self.declare_parameter('max_angular_speed', 1.2)
        self.declare_parameter('kp_angular', 2.5)

        # --- State Variables ---
        self.map_info, self.grid = None, None
        self.current_pose, self.current_yaw = None, None
        self.path_to_follow = None
        self.path_index = 0

        # --- NEW: State variables to manage dynamic start and planning trigger ---
        self.start_point_world = None
        self.map_received = False
        self.initial_pose_received = False
        self.planning_complete = False

        # --- ROS Subscriptions & Publishers ---
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_publisher = self.create_publisher(Path, '/global_plan', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("Node initialized. Waiting for map and initial robot pose...")

    def odom_callback(self, msg):
        # This part runs on every message to update the robot's current state
        self.current_pose = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        t3 = +2.0 * (q[3] * q[2] + q[0] * q[1])
        t4 = +1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
        self.current_yaw = math.atan2(t3, t4)

        # --- NEW: This part runs only ONCE to capture the initial pose ---
        if not self.initial_pose_received:
            self.start_point_world = [self.current_pose.x, self.current_pose.y]
            self.initial_pose_received = True
            self.get_logger().info(f"Dynamic start point received from odometry: {self.start_point_world}")
            self._trigger_planning_if_ready()

    def map_callback(self, msg):
        if self.map_received:
            return # Avoid reprocessing the map
        
        self.map_info = msg.info
        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_received = True
        self.get_logger().info("Map received.")
        self._trigger_planning_if_ready()
    
    def _trigger_planning_if_ready(self):
        """Checks if all conditions are met to start A* planning."""
        if self.map_received and self.initial_pose_received and not self.planning_complete:
            self.planning_complete = True # Ensure this block runs only once
            self._execute_planning()

    def _execute_planning(self):
        """Contains the main planning logic, now separated for clarity."""
        self.get_logger().info("Map and initial pose are ready. Starting A* planning...")
        
        goal_w = self.get_parameter('goal_point').value
        start_grid = self.world_to_grid(self.start_point_world)
        goal_grid = self.world_to_grid(goal_w)
        
        self.get_logger().info(f"Planning from {self.start_point_world} (grid: {start_grid}) to {goal_w} (grid: {goal_grid})")

        path_grid, explored_grid = a_star_search(self.grid, start_grid, goal_grid)

        self.plot_graph(start_grid, goal_grid, path_grid, explored_grid)

        if path_grid:
            self.get_logger().info("Path found. Preparing to execute.")
            self.path_to_follow = [self.grid_to_world(p) for p in path_grid]
            self.publish_path_viz(self.path_to_follow)
            self.controller_timer = self.create_timer(0.1, self.control_loop)
            self.get_logger().info("Controller started.")
        else:
            self.get_logger().error("No path could be found! Cannot move the robot.")

    def control_loop(self):
        if self.path_to_follow is None or self.current_pose is None or self.current_yaw is None:
            return

        robot_pos = np.array([self.current_pose.x, self.current_pose.y])
        path_points = np.array(self.path_to_follow)
        
        distances = np.linalg.norm(path_points[self.path_index:] - robot_pos, axis=1)
        relative_closest_index = np.argmin(distances)
        self.path_index += relative_closest_index

        lookahead_dist = self.get_parameter('lookahead_distance').value
        target_index = self.path_index
        while target_index < len(path_points) - 1:
            dist_to_target = np.linalg.norm(path_points[target_index] - robot_pos)
            if dist_to_target >= lookahead_dist:
                break
            target_index += 1
        
        if np.linalg.norm(path_points[-1] - robot_pos) < lookahead_dist:
            target_index = len(path_points) - 1

        target_pos = path_points[target_index]
        
        # dist_to_final_goal = np.linalg.norm(path_points[-1] - robot_pos)
        

        angle_to_target = math.atan2(target_pos[1] - self.current_pose.y, target_pos[0] - self.current_pose.x)
        heading_error = angle_to_target - self.current_yaw
        if heading_error > math.pi: heading_error -= 2 * math.pi
        if heading_error < -math.pi: heading_error += 2 * math.pi

        kp_angular = self.get_parameter('kp_angular').value
        max_linear = self.get_parameter('max_linear_speed').value
        max_angular = self.get_parameter('max_angular_speed').value
        angular_vel = kp_angular * heading_error
        scaling_factor = max(0.3, 1.0 - 0.9 * abs(heading_error) / (math.pi/2))
        linear_vel = max_linear * scaling_factor
        
        total_waypoints = len(self.path_to_follow)
        completed_waypoints = self.path_index + 1
        remaining_waypoints = total_waypoints - completed_waypoints
        if remaining_waypoints<1:
            total_waypoints = len(self.path_to_follow)
            completed_waypoints = self.path_index + 1
            if completed_waypoints > total_waypoints: completed_waypoints = total_waypoints
            
            percentage_complete = (completed_waypoints / total_waypoints) * 100 if total_waypoints > 0 else 100

            self.get_logger().info(
                f"Goal Reached! Path progress: {completed_waypoints}/{total_waypoints} "
                f"waypoints covered ({percentage_complete:.1f}%). Robot stopped."
            )
            
            self.controller_timer.cancel()
            self.stop_robot()
            return
        if total_waypoints > 0:
            percentage_complete = (completed_waypoints / total_waypoints) * 100
            self.get_logger().info(
                f"Progress: [{completed_waypoints}/{total_waypoints}] waypoints | "
                f"Completed: {percentage_complete:.1f}% | "
                f"Remaining: {remaining_waypoints} waypoints.",
                throttle_duration_sec=2.0
            )
        
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = max(-max_angular, min(max_angular, angular_vel))
        self.cmd_vel_publisher.publish(twist_msg)

    def stop_robot(self):
        self.get_logger().info("Sending zero-velocity command.")
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(stop_msg)

    def world_to_grid(self, wc): 
        return (int((wc[1] - self.map_info.origin.position.y) / self.map_info.resolution), 
                int((wc[0] - self.map_info.origin.position.x) / self.map_info.resolution))

    def grid_to_world(self, gc): 
        return [(gc[1] + 0.5) * self.map_info.resolution + self.map_info.origin.position.x, 
                (gc[0] + 0.5) * self.map_info.resolution + self.map_info.origin.position.y]
    
    def publish_path_viz(self, path_world):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for point in path_world:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x, pose_stamped.pose.position.y = point[0], point[1]
            pose_stamped.pose.orientation.w = 1.0 
            path_msg.poses.append(pose_stamped)
        self.path_publisher.publish(path_msg)

    def plot_graph(self, start, end, path, explored_path):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_title("A* Global Plan")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.imshow(self.grid, cmap='gray_r', interpolation='none', origin='lower', 
                  extent=[self.map_info.origin.position.x, 
                          self.map_info.origin.position.x + self.map_info.width * self.map_info.resolution,
                          self.map_info.origin.position.y,
                          self.map_info.origin.position.y + self.map_info.height * self.map_info.resolution])
        def plot_grid_points(points, color, alpha, size, label):
            if not points: return
            world_points = np.array([self.grid_to_world(p) for p in points])
            ax.scatter(world_points[:, 0], world_points[:, 1], c=color, alpha=alpha, s=size, label=label)
        plot_grid_points(explored_path, 'yellow', 0.2, 10, 'Explored Cells')
        if path:
            path_world = [self.grid_to_world(p) for p in path]
            path_x, path_y = zip(*path_world)
            ax.plot(path_x, path_y, color='blue', linewidth=2, marker='o', markersize=3, label='Final Path')
        start_w, end_w = self.grid_to_world(start), self.grid_to_world(end)
        ax.plot(start_w[0], start_w[1], 'go', markersize=12, label='Start', markeredgecolor='k')
        ax.plot(end_w[0], end_w[1], 'ro', markersize=12, label='Goal', markeredgecolor='k')
        ax.legend()
        ax.grid(True)

        dir = 'astar_output'

        # Check if directory exists, if not create it
        if not os.path.exists(dir):
            os.makedirs(dir)
        file_name = f"{dir}/astar_path_plan.png"
        plt.savefig(file_name)
        plt.close(fig)
        self.get_logger().info(f"Plot saved to '{file_name}'")

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerAndControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down A* planner node.")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()