#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import os

# ROS 2 message types
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile, DurabilityPolicy

class PyBulletRobotSimulator(Node):
    """
    A self-contained ROS 2 Node that simulates a robot in a custom environment
    and publishes all necessary data, including a static map of the world.
    """
    def __init__(self):
        super().__init__('pybullet_robot_simulator_node')
        self.get_logger().info('PyBullet Robot Simulator Node started.')

        # Initialize PyBullet in GUI mode
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Configure simulation time step
        self.sim_frequency_hz = 50.0
        self.time_step = 1.0 / self.sim_frequency_hz
        p.setTimeStep(self.time_step)

        # Load robot and environment
        self.robot_id = p.loadURDF("r2d2.urdf", [4.0, 4.0, 0.1], useFixedBase=False)
        self.get_logger().info(f"Robot loaded with ID: {self.robot_id}")
        self.plane_id = p.loadURDF("plane.urdf")
        self.wall_ids = self.create_simulation_environment()

        # --- Robot control parameters ---
        self.left_front_wheel_joint_index = 6
        self.left_rear_wheel_joint_index = 7
        self.right_front_wheel_joint_index = 2
        self.right_rear_wheel_joint_index = 3
        self.wheel_radius = 0.05
        self.wheel_base = 0.2

        # Log joint names for debugging purposes
        self.get_logger().info("--- Joint Information ---")
        for joint_id in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, joint_id)
            self.get_logger().info(f"Joint {joint_id}: Name={joint_info[1].decode('utf-8')}, Type={joint_info[2]}")
        self.get_logger().info("-------------------------")

        # --- ROS 2 Publishers and Subscribers ---
        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.odom_publisher = self.create_publisher(Odometry, '/odom', 10)
        self.laser_publisher = self.create_publisher(LaserScan, '/scan', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # <<< ADDED >>> Map publisher with "latched" QoS profile
        map_qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', map_qos_profile)

        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = 'odom'
        self.odom_msg.child_frame_id = 'base_link'
        self.last_robot_pos = np.array([0.0, 0.0, 0.0])

        # --- Laser scan parameters ---
        self.num_laser_beams = 180
        self.laser_range_max = 15.0
        self.laser_angle_min = -np.pi
        self.laser_angle_max = np.pi
        self.laser_angle_increment = (self.laser_angle_max - self.laser_angle_min) / self.num_laser_beams
        self.laser_height = 0.6
        self.debug_draw_laser = True

        self.publish_static_transforms()

        # <<< ADDED >>> Create and publish the map once on startup
        self.create_and_publish_map()

        # --- Command velocities ---
        self.cmd_linear_vel = 0.0
        self.cmd_angular_vel = 0.0

        # --- Timers ---
        self.sim_timer = self.create_timer(self.time_step, self.simulation_step)

    def create_simulation_environment(self):
        """
        Creates the walls and obstacles in the PyBullet simulation.
        """
        self.get_logger().info("Creating simulation environment...")
        wall_ids = []
        wall_half_length = 10.0
        wall_height = 1.5
        wall_thickness = 0.1

        # Outer Walls
        wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_half_length, wall_thickness, wall_height/2.0])
        wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_half_length, wall_thickness, wall_height/2.0], rgbaColor=[0.8, 0.2, 0.2, 1])
        wall_ids.append(p.createMultiBody(0, wall_col, wall_vis, [0, wall_half_length, wall_height/2.0]))
        wall_ids.append(p.createMultiBody(0, wall_col, wall_vis, [0, -wall_half_length, wall_height/2.0]))

        wall_side_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, wall_half_length, wall_height/2.0])
        wall_side_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness, wall_half_length, wall_height/2.0], rgbaColor=[0.8, 0.2, 0.2, 1])
        wall_ids.append(p.createMultiBody(0, wall_side_col, wall_side_vis, [wall_half_length, 0, wall_height/2.0]))
        wall_ids.append(p.createMultiBody(0, wall_side_col, wall_side_vis, [-wall_half_length, 0, wall_height/2.0]))

        # Central Box
        center_box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.0, 1.0, 1.0/2.0])
        center_box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.0, 1.0, 1.0/2.0], rgbaColor=[0.2, 0.8, 0.2, 1])
        wall_ids.append(p.createMultiBody(0, center_box_col, center_box_vis, [0, 0, 1.0/2.0]))

        # 4 Cylinders
        cyl_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.5, height=wall_height)
        cyl_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=wall_height, rgbaColor=[0.5, 0.5, 0.5, 1])
        wall_ids.append(p.createMultiBody(0, cyl_col, cyl_vis, [-6, 0, wall_height / 2.0]))
        wall_ids.append(p.createMultiBody(0, cyl_col, cyl_vis, [6, 0, wall_height / 2.0]))
        wall_ids.append(p.createMultiBody(0, cyl_col, cyl_vis, [0, 6, wall_height / 2.0]))
        wall_ids.append(p.createMultiBody(0, cyl_col, cyl_vis, [0, -6, wall_height / 2.0]))

        # 4 Corner Boxes
        corner_box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, wall_height/2.0])
        corner_box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, wall_height/2.0], rgbaColor=[0.2, 0.2, 0.8, 1])
        wall_ids.append(p.createMultiBody(0, corner_box_col, corner_box_vis, [6, 6, wall_height / 2.0]))
        wall_ids.append(p.createMultiBody(0, corner_box_col, corner_box_vis, [-6, 6, wall_height / 2.0]))
        wall_ids.append(p.createMultiBody(0, corner_box_col, corner_box_vis, [6, -6, wall_height / 2.0]))
        wall_ids.append(p.createMultiBody(0, corner_box_col, corner_box_vis, [-6, -6, wall_height / 2.0]))

        return wall_ids

    def create_and_publish_map(self):
        """
        Generates an OccupancyGrid from the PyBullet world and publishes it once.
        """
        self.get_logger().info("Creating and publishing occupancy grid map...")
        map_resolution = 0.1
        map_width_m = 22.0
        map_height_m = 22.0
        map_size_x = int(map_width_m / map_resolution)
        map_size_y = int(map_height_m / map_resolution)
        map_origin_x = -map_width_m / 2.0
        map_origin_y = -map_height_m / 2.0

        grid = np.zeros((map_size_y, map_size_x), dtype=np.int8)

        def world_to_grid(wx, wy):
            gx = int((wx - map_origin_x) / map_resolution)
            gy = int((wy - map_origin_y) / map_resolution)
            return gx, gy

        # Outer Walls
        x_start, y_start = world_to_grid(-10.1, -10.1)
        x_end, y_end = world_to_grid(10.1, 10.1)
        grid[y_start:y_end, x_start:x_start+2] = 100
        grid[y_start:y_end, x_end-2:x_end] = 100
        grid[y_start:y_start+2, x_start:x_end] = 100
        grid[y_end-2:y_end, x_start:x_end] = 100

        # Central Box
        x_start, y_start = world_to_grid(-1.0, -1.0)
        x_end, y_end = world_to_grid(1.0, 1.0)
        grid[y_start:y_end, x_start:x_end] = 100

        def draw_cylinder(cx, cy, r):
            for i in range(map_size_x):
                for j in range(map_size_y):
                    wx = map_origin_x + (i + 0.5) * map_resolution
                    wy = map_origin_y + (j + 0.5) * map_resolution
                    if (wx - cx)**2 + (wy - cy)**2 < r**2:
                        grid[j, i] = 100
        
        draw_cylinder(-6, 0, 0.5)
        draw_cylinder(6, 0, 0.5)
        draw_cylinder(0, 6, 0.5)
        draw_cylinder(0, -6, 0.5)

        def draw_box(cx, cy, h_size):
            x_start, y_start = world_to_grid(cx - h_size, cy - h_size)
            x_end, y_end = world_to_grid(cx + h_size, cy + h_size)
            grid[y_start:y_end, x_start:x_end] = 100

        draw_box(6, 6, 0.5)
        draw_box(-6, 6, 0.5)
        draw_box(6, -6, 0.5)
        draw_box(-6, -6, 0.5)

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.map_load_time = msg.header.stamp
        msg.info.resolution = map_resolution
        msg.info.width = map_size_x
        msg.info.height = map_size_y
        msg.info.origin.position.x = map_origin_x
        msg.info.origin.position.y = map_origin_y
        msg.data = grid.flatten().tolist()
        # print(msg)
        self.map_publisher.publish(msg)
        self.get_logger().info("Map published successfully.")

    def publish_static_transforms(self):
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = 'base_link'
        static_transform.child_frame_id = 'laser_frame'
        static_transform.transform.translation.z = self.laser_height
        static_transform.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(static_transform)

    def cmd_vel_callback(self, msg):
        self.cmd_linear_vel = msg.linear.x
        self.cmd_angular_vel = msg.angular.z

    def apply_robot_control(self, linear_vel, angular_vel):
        left_wheel_vel = (linear_vel - angular_vel * self.wheel_base / 2.0) / self.wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * self.wheel_base / 2.0) / self.wheel_radius

        p.setJointMotorControl2(self.robot_id, self.left_front_wheel_joint_index, p.VELOCITY_CONTROL, targetVelocity=left_wheel_vel, force=500)
        p.setJointMotorControl2(self.robot_id, self.left_rear_wheel_joint_index, p.VELOCITY_CONTROL, targetVelocity=left_wheel_vel, force=500)
        p.setJointMotorControl2(self.robot_id, self.right_front_wheel_joint_index, p.VELOCITY_CONTROL, targetVelocity=right_wheel_vel, force=500)
        p.setJointMotorControl2(self.robot_id, self.right_rear_wheel_joint_index, p.VELOCITY_CONTROL, targetVelocity=right_wheel_vel, force=500)

    def simulation_step(self):
        self.apply_robot_control(self.cmd_linear_vel, self.cmd_angular_vel)
        p.stepSimulation()
        self.publish_odometry_and_tf()
        self.simulate_and_publish_laser_scan()

    def simulate_and_publish_laser_scan(self):
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot_id)
        _, _, robot_yaw = p.getEulerFromQuaternion(robot_ori)
        laser_origin_world = [robot_pos[0], robot_pos[1], robot_pos[2] + self.laser_height]

        ray_starts = [laser_origin_world] * self.num_laser_beams
        ray_ends = []
        for i in range(self.num_laser_beams):
            angle = self.laser_angle_min + i * self.laser_angle_increment
            world_angle = robot_yaw + angle
            ray_ends.append([
                laser_origin_world[0] + self.laser_range_max * math.cos(world_angle),
                laser_origin_world[1] + self.laser_range_max * math.sin(world_angle),
                laser_origin_world[2]
            ])

        results = p.rayTestBatch(ray_starts, ray_ends)

        laser_msg = LaserScan()
        laser_msg.header.stamp = self.get_clock().now().to_msg()
        laser_msg.header.frame_id = 'laser_frame'
        laser_msg.angle_min = self.laser_angle_min
        laser_msg.angle_max = self.laser_angle_max
        laser_msg.angle_increment = self.laser_angle_increment
        laser_msg.range_min = 0.1
        laser_msg.range_max = self.laser_range_max

        ranges = []
        for result in results:
            hit_fraction = result[2]
            if hit_fraction == 1.0:
                ranges.append(float('inf'))
            else:
                distance = hit_fraction * self.laser_range_max
                ranges.append(distance)

        laser_msg.ranges = ranges
        self.laser_publisher.publish(laser_msg)

    def publish_odometry_and_tf(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        current_ros_time = self.get_clock().now()

        self.odom_msg.header.stamp = current_ros_time.to_msg()
        self.odom_msg.pose.pose.position.x = pos[0]
        self.odom_msg.pose.pose.position.y = pos[1]
        self.odom_msg.pose.pose.position.z = pos[2]
        self.odom_msg.pose.pose.orientation.x = ori[0]
        self.odom_msg.pose.pose.orientation.y = ori[1]
        self.odom_msg.pose.pose.orientation.z = ori[2]
        self.odom_msg.pose.pose.orientation.w = ori[3]
        self.odom_publisher.publish(self.odom_msg)

        t = TransformStamped()
        t.header.stamp = current_ros_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        t.transform.rotation.x = ori[0]
        t.transform.rotation.y = ori[1]
        t.transform.rotation.z = ori[2]
        t.transform.rotation.w = ori[3]
        self.tf_broadcaster.sendTransform(t)

    def destroy_node(self):
        p.disconnect()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PyBulletRobotSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()