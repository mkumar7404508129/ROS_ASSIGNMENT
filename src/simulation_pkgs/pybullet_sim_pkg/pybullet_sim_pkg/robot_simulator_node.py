#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pybullet as p
import pybullet_data
import time
import numpy as np
import math

# ROS 2 message types
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

class PyBulletRobotSimulator(Node):
    """
    An updated ROS 2 Node that interfaces a PyBullet simulation with ROS 2 topics.
    This version acts as a pure simulator/driver, subscribing to Twist messages
    from a separate controller node and publishing Odometry and LaserScan data.
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
        # The robot is placed in a corner to ensure rays hit the walls for debugging.
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
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)

        self.odom_publisher = self.create_publisher(Odometry, '/odom', 10)
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = 'odom'
        self.odom_msg.child_frame_id = 'base_link'
        self.odom_msg.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_robot_pos = np.array([0.0, 0.0, 0.0])

        # Initialize the static TF broadcaster only once
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.laser_height = 0.6
        # Create and publish the static transform from base_link to laser_frame
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = 'base_link'
        static_transform.child_frame_id = 'laser_frame'
        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = self.laser_height
        static_transform.transform.rotation.x = 0.0
        static_transform.transform.rotation.y = 0.0
        static_transform.transform.rotation.z = 0.0
        static_transform.transform.rotation.w = 1.0

        self.static_tf_broadcaster.sendTransform(static_transform)

        self.laser_publisher = self.create_publisher(LaserScan, '/scan', 10)
        self.laser_msg = LaserScan()
        self.laser_msg.header.frame_id = 'laser_frame'
        
        # --- Laser scan parameters ---
        self.num_laser_beams = 180
        self.laser_range_max = 10.0
        self.laser_angle_min = -np.pi / 2.0  
        self.laser_angle_max = np.pi / 2.0   
        self.laser_angle_increment = (self.laser_angle_max - self.laser_angle_min) / self.num_laser_beams
        self.debug_draw_laser = False

        # --- Command velocities from the controller node ---
        self.cmd_linear_vel = 0.0
        self.cmd_angular_vel = 0.0

        # --- Timers for simulation and publishing ---
        self.sim_timer = self.create_timer(self.time_step, self.simulation_step)
        
    def create_simulation_environment(self):
        """
        Creates the walls and obstacles in the PyBullet simulation with increased height.
        """
        wall_ids = []
        wall_half_length = 10.0 
        wall_height = 1.5 
        wall_thickness = 0.1
        
        # Create a red box visual and collision shape for the walls
        wall_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_half_length, wall_thickness, wall_height])
        wall_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_half_length, wall_thickness, wall_height], rgbaColor=[0.8, 0.2, 0.2, 1])
        
        # Top and bottom walls
        wall_ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, wall_half_length, wall_height]))
        wall_ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, -wall_half_length, wall_height]))

        # Create a separate shape for the side walls
        wall_side_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, wall_half_length, wall_height])
        wall_side_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness, wall_half_length, wall_height], rgbaColor=[0.8, 0.2, 0.2, 1])
        
        # Left and right walls
        wall_ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_side_collision_shape, baseVisualShapeIndex=wall_side_visual_shape, basePosition=[wall_half_length, 0, wall_height]))
        wall_ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_side_collision_shape, baseVisualShapeIndex=wall_side_visual_shape, basePosition=[-wall_half_length, 0, wall_height]))

        # Create a green obstacle in the center
        obs_half_size = 2.0 
        obs_height = 1.5 
        obs_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[obs_half_size, obs_half_size, obs_height])
        obs_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[obs_half_size, obs_half_size, obs_height], rgbaColor=[0.2, 0.8, 0.2, 1])
        wall_ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obs_collision_shape, baseVisualShapeIndex=obs_visual_shape, basePosition=[0, 0, obs_height]))
        
        return wall_ids

    def cmd_vel_callback(self, msg):
        """
        Callback for receiving Twist messages from the controller node.
        Updates the target linear and angular velocities.
        """
        self.cmd_linear_vel = msg.linear.x
        self.cmd_angular_vel = msg.angular.z

    def apply_robot_control(self, linear_vel, angular_vel):
        """
        Applies velocity commands to the robot's wheels.
        """
        left_wheel_vel = (linear_vel - angular_vel * self.wheel_base / 2.0) / self.wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * self.wheel_base / 2.0) / self.wheel_radius

        p.setJointMotorControl2(self.robot_id, self.left_front_wheel_joint_index,
                                p.VELOCITY_CONTROL, targetVelocity=left_wheel_vel, force=500)
        p.setJointMotorControl2(self.robot_id, self.left_rear_wheel_joint_index,
                                p.VELOCITY_CONTROL, targetVelocity=left_wheel_vel, force=500)
        p.setJointMotorControl2(self.robot_id, self.right_front_wheel_joint_index,
                                p.VELOCITY_CONTROL, targetVelocity=right_wheel_vel, force=500)
        p.setJointMotorControl2(self.robot_id, self.right_rear_wheel_joint_index,
                                p.VELOCITY_CONTROL, targetVelocity=right_wheel_vel, force=500)

    def simulation_step(self):
        """
        Main simulation loop function.
        Applies commands, steps the simulation, and publishes sensor data.
        """
        # 1. Apply the most recent commands to the robot
        self.apply_robot_control(self.cmd_linear_vel, self.cmd_angular_vel)
        
        # 2. Step the PyBullet simulation
        p.stepSimulation()
        
        # 3. Publish odometry and TF
        self.publish_odometry_and_tf()

        # 4. Simulate the laser scan and publish the result
        self.simulate_and_publish_laser_scan()
        
    def simulate_and_publish_laser_scan(self):
        """
        Simulates laser scan rays in PyBullet, populates a LaserScan message,
        and publishes it.
        """
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot_id)
        _, _, robot_yaw = p.getEulerFromQuaternion(robot_ori)

        laser_origin_world = [robot_pos[0], robot_pos[1], robot_pos[2] + self.laser_height]

        ray_starts = [laser_origin_world] * self.num_laser_beams
        ray_ends = []
        for i in range(self.num_laser_beams):
            angle = self.laser_angle_min + i * self.laser_angle_increment
            world_angle = robot_yaw + angle
            ray_end = [
                laser_origin_world[0] + self.laser_range_max * math.cos(world_angle),
                laser_origin_world[1] + self.laser_range_max * math.sin(world_angle),
                laser_origin_world[2]
            ]
            ray_ends.append(ray_end)

        results = p.rayTestBatch(ray_starts, ray_ends)

        self.laser_msg.header.stamp = self.get_clock().now().to_msg()
        self.laser_msg.angle_min = self.laser_angle_min
        self.laser_msg.angle_max = self.laser_angle_max
        self.laser_msg.angle_increment = self.laser_angle_increment
        self.laser_msg.time_increment = 0.0
        self.laser_msg.range_min = 0.1
        self.laser_msg.range_max = self.laser_range_max
        self.laser_msg.ranges = [float('inf')] * self.num_laser_beams
        
        for i, result in enumerate(results):
            hit_fraction = result[0]
            
            # hit_fraction is between 0.0 (hit at origin) and 1.0 (no hit)
            if hit_fraction > 0.0:
                # Correctly calculate the distance by multiplying the fraction by the max range
                distance = hit_fraction 
                if self.laser_msg.range_min <= distance <= self.laser_msg.range_max:
                    self.laser_msg.ranges[i] = distance
            
            if self.debug_draw_laser:
                if hit_fraction > 0.0:
                    hit_position = result[3]
                    color = [0, 1, 0] # Green for a hit
                    p.addUserDebugLine(laser_origin_world, hit_position, lineColorRGB=color, lifeTime=self.time_step * 2)
                else:
                    color = [1, 0, 0] # Red for no hit
                    p.addUserDebugLine(laser_origin_world, ray_ends[i], lineColorRGB=color, lifeTime=self.time_step * 2)
        
        self.laser_publisher.publish(self.laser_msg)

    def publish_odometry_and_tf(self):
        """
        Calculates and publishes robot's odometry and TF transform.
        """
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        _, _, yaw = p.getEulerFromQuaternion(ori)
        current_robot_pos = np.array([pos[0], pos[1], yaw])
        current_ros_time = self.get_clock().now()
        
        last_odom_time_rclpy = rclpy.time.Time(
            seconds=self.odom_msg.header.stamp.sec,
            nanoseconds=self.odom_msg.header.stamp.nanosec,
            clock_type=self.get_clock().clock_type
        )
        dt = (current_ros_time - last_odom_time_rclpy).nanoseconds / 1e9

        if dt == 0.0:
            dt = self.time_step

        prev_pos = self.last_robot_pos
        self.last_robot_pos = current_robot_pos
            
        lin_vel_x = (current_robot_pos[0] - prev_pos[0]) / dt
        lin_vel_y = (current_robot_pos[1] - prev_pos[1]) / dt
        delta_yaw = current_robot_pos[2] - prev_pos[2]
        if delta_yaw > math.pi:
            delta_yaw -= 2 * math.pi
        elif delta_yaw < -math.pi:
            delta_yaw += 2 * math.pi
        ang_vel_z = delta_yaw / dt

        self.odom_msg.header.stamp = current_ros_time.to_msg()
        self.odom_msg.pose.pose.position.x = pos[0]
        self.odom_msg.pose.pose.position.y = pos[1]
        self.odom_msg.pose.pose.position.z = pos[2]
        self.odom_msg.pose.pose.orientation.x = ori[0]
        self.odom_msg.pose.pose.orientation.y = ori[1]
        self.odom_msg.pose.pose.orientation.z = ori[2]
        self.odom_msg.pose.pose.orientation.w = ori[3]

        self.odom_msg.twist.twist.linear.x = lin_vel_x
        self.odom_msg.twist.twist.linear.y = lin_vel_y
        self.odom_msg.twist.twist.linear.z = 0.0
        self.odom_msg.twist.twist.angular.x = 0.0
        self.odom_msg.twist.twist.angular.y = 0.0
        self.odom_msg.twist.twist.angular.z = ang_vel_z

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
        """
        Clean up PyBullet connection when the node is destroyed.
        """
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