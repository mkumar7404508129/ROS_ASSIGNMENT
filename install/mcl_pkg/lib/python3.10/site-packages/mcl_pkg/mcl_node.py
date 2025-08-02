#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration

import numpy as np
from numpy.random import uniform, randn
import math
from scipy.stats import norm

# ROS 2 message types
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseArray, TransformStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time as TimeMsg

class MCLNode(Node):
    def __init__(self):
        super().__init__('mcl_node')
        self.get_logger().info('Monte Carlo Localization Node has been started.')

        # --- Parameters ---
        self.num_particles = self.declare_parameter('num_particles', 100).value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.laser_frame = self.declare_parameter('laser_frame', 'laser_frame').value

        # Motion model noise parameters (adjust for your robot)
        self.alpha1 = self.declare_parameter('alpha1', 0.1).value  # Rotational noise from rotational movement
        self.alpha2 = self.declare_parameter('alpha2', 0.1).value  # Rotational noise from translational movement
        self.alpha3 = self.declare_parameter('alpha3', 0.1).value  # Translational noise from translational movement
        self.alpha4 = self.declare_parameter('alpha4', 0.1).value  # Translational noise from rotational movement

        # Sensor model noise parameters (adjust for your sensor)
        # These weights should sum to 1.0
        self.sigma_hit = self.declare_parameter('sigma_hit', 0.1).value
        self.lambda_short = self.declare_parameter('lambda_short', 0.1).value
        self.z_hit = self.declare_parameter('z_hit', 0.5).value
        self.z_short = self.declare_parameter('z_short', 0.2).value
        self.z_max = self.declare_parameter('z_max', 0.2).value
        self.z_rand = self.declare_parameter('z_rand', 0.1).value

        # --- Map ---
        self.occupancy_grid = self.create_occupancy_grid()
        self.get_logger().info('Occupancy grid map created.')

        # --- State variables ---
        self.particles = np.zeros((self.num_particles, 3))  # [x, y, yaw]
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.last_odom_pose = None
        self.estimated_pose = np.zeros(3)  # [x, y, yaw]

        # --- ROS 2 Publishers and Subscribers ---
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initial_pose_callback, 10)

        self.pose_pub = self.create_publisher(PoseStamped, '/amcl_pose', 10)
        self.particle_pub = self.create_publisher(PoseArray, '/particlecloud', 10)
        
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize particles
        self.initialize_particles()
    
    def create_occupancy_grid(self):
        """Creates a simple hardcoded occupancy grid for this example."""
        # Represents a 10x10 meter map with a square obstacle in the middle
        map_size = 100
        grid = np.ones((map_size, map_size), dtype=np.int8) * -1  # -1 for unknown
        
        # Assume a 10m x 10m map with 0.1m/cell resolution
        map_resolution = 0.1
        map_origin_x = -5.0
        map_origin_y = -5.0

        # Define map walls (0 for free, 100 for occupied)
        grid[:, 0] = 100
        grid[:, map_size - 1] = 100
        grid[0, :] = 100
        grid[map_size - 1, :] = 100

        # Create a central obstacle
        obs_start = int(map_size / 2) - 10
        obs_end = int(map_size / 2) + 10
        grid[obs_start:obs_end, obs_start:obs_end] = 100

        return {
            'data': grid,
            'info': {
                'resolution': map_resolution,
                'width': map_size,
                'height': map_size,
                'origin': {'position': {'x': map_origin_x, 'y': map_origin_y, 'z': 0.0}}
            }
        }

    def world_to_map(self, x, y):
        """Converts world coordinates to map grid indices."""
        info = self.occupancy_grid['info']
        map_x = int((x - info['origin']['position']['x']) / info['resolution'])
        map_y = int((y - info['origin']['position']['y']) / info['resolution'])
        return map_x, map_y

    def is_occupied(self, x, y):
        """Checks if a map cell is occupied."""
        map_x, map_y = self.world_to_map(x, y)
        info = self.occupancy_grid['info']
        if 0 <= map_x < info['width'] and 0 <= map_y < info['height']:
            # The OccupancyGrid message stores data as a 1D array.
            # Your numpy array is 2D, so you need to be careful with indexing.
            # A simple 2D numpy array is ok for this example.
            return self.occupancy_grid['data'][map_y, map_x] > 50  # 50 is common threshold
        return True # Assume out of bounds is occupied

    def ray_cast_map(self, x, y, yaw, angle, max_range):
        """Simulates a laser ray cast against the internal map."""
        step_size = self.occupancy_grid['info']['resolution'] / 2.0
        
        current_x, current_y = x, y
        distance = 0.0

        # Calculate ray direction
        ray_yaw = yaw + angle
        dx = step_size * math.cos(ray_yaw)
        dy = step_size * math.sin(ray_yaw)

        while distance < max_range:
            if self.is_occupied(current_x, current_y):
                return distance
            
            current_x += dx
            current_y += dy
            distance += step_size
            
        return max_range


    def initialize_particles(self):
        """Initializes particles randomly across the map."""
        self.get_logger().info(f"Initializing {self.num_particles} particles.")
        info = self.occupancy_grid['info']
        min_x = info['origin']['position']['x']
        max_x = min_x + info['width'] * info['resolution']
        min_y = info['origin']['position']['y']
        max_y = min_y + info['height'] * info['resolution']

        for i in range(self.num_particles):
            self.particles[i, 0] = uniform(min_x, max_x)
            self.particles[i, 1] = uniform(min_y, max_y)
            self.particles[i, 2] = uniform(-math.pi, math.pi)
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.publish_particles()


    def initial_pose_callback(self, msg):
        """Handles initial pose from Rviz2, re-initializing particles around it."""
        self.get_logger().info("Received initialpose, re-initializing particles.")
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quaternion = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(quaternion)
        
        # Spread particles around the initial pose
        for i in range(self.num_particles):
            self.particles[i, 0] = x + np.random.normal(0, 0.5) # Add noise
            self.particles[i, 1] = y + np.random.normal(0, 0.5)
            self.particles[i, 2] = yaw + np.random.normal(0, 0.2)
        
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.publish_particles()


    def odom_callback(self, msg):
        """Motion model update."""
        if self.last_odom_pose is None:
            self.last_odom_pose = msg.pose.pose
            return

        current_pose = msg.pose.pose
        last_pose = self.last_odom_pose
        self.last_odom_pose = current_pose

        current_yaw = self.quaternion_to_yaw(current_pose.orientation)
        last_yaw = self.quaternion_to_yaw(last_pose.orientation)

        delta_rot1 = math.atan2(current_pose.position.y - last_pose.position.y,
                                current_pose.position.x - last_pose.position.x) - last_yaw
        delta_trans = math.sqrt((current_pose.position.x - last_pose.position.x)**2 +
                                (current_pose.position.y - last_pose.position.y)**2)
        delta_rot2 = current_yaw - last_yaw - delta_rot1

        for i in range(self.num_particles):
            rot1_noise = randn() * (self.alpha1 * abs(delta_rot1) + self.alpha2 * abs(delta_trans))
            trans_noise = randn() * (self.alpha3 * abs(delta_trans) + self.alpha4 * abs(delta_rot1 + delta_rot2))
            rot2_noise = randn() * (self.alpha1 * abs(delta_rot2) + self.alpha2 * abs(delta_trans))

            self.particles[i, 0] += (delta_trans + trans_noise) * math.cos(self.particles[i, 2] + delta_rot1 + rot1_noise)
            self.particles[i, 1] += (delta_trans + trans_noise) * math.sin(self.particles[i, 2] + delta_rot1 + rot1_noise)
            self.particles[i, 2] += delta_rot1 + rot1_noise + delta_rot2 + rot2_noise

    def laser_callback(self, msg):
        """Measurement model update and resampling."""
        if self.last_odom_pose is None:
            return
        
        self.get_logger().info('Received laser scan. Updating particle weights.')

        # Calculate new weights for each particle
        for i in range(self.num_particles):
            self.weights[i] = self.sensor_model(msg, self.particles[i])

        # Normalize weights
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights /= total_weight
        else:
            self.get_logger().warn("All particles have zero weight! Re-initializing.")
            self.initialize_particles()
            return
        
        # Resampling
        self.resample()

        # Update and publish estimated pose
        self.estimate_pose()
        self.publish_particles()


    def sensor_model(self, laser_scan, particle_pose):
        """Calculates the likelihood of a particle's pose given a laser scan
           using a full four-component sensor model.
           
           This version uses log-likelihoods to prevent floating-point underflow
           when multiplying many small probabilities.
        """
        particle_x, particle_y, particle_yaw = particle_pose
        total_log_likelihood = 0.0
        
        # Iterate through a subset of laser beams for efficiency
        step = 10
        for i in range(0, len(laser_scan.ranges), step):
            actual_range = laser_scan.ranges[i]
            
            # Filter out invalid readings
            if np.isinf(actual_range) or np.isnan(actual_range):
                actual_range = laser_scan.range_max

            angle = laser_scan.angle_min + i * laser_scan.angle_increment
            expected_range = self.ray_cast_map(particle_x, particle_y, particle_yaw, angle, laser_scan.range_max)
            
            # Calculate probabilities for the four models
            p_hit = 0.0
            if 0.0 < actual_range < laser_scan.range_max:
                # Hit model: normal distribution around the expected range
                p_hit = norm.pdf(actual_range, expected_range, self.sigma_hit)
                
            p_short = 0.0
            if 0.0 < actual_range < expected_range:
                # Short model: exponential distribution
                # The PDF of an exponential distribution is f(x; lambda) = lambda * exp(-lambda * x) for x >= 0
                p_short = self.lambda_short * math.exp(-self.lambda_short * actual_range)
                
            p_max = 0.0
            if actual_range >= laser_scan.range_max:
                # Max range model
                if abs(expected_range - laser_scan.range_max) < 0.1:
                    p_max = 1.0 # High likelihood if expected range is also max range
            else: # If the actual range is not max, this component contributes nothing
                p_max = 0.0

            p_rand = 1.0 / laser_scan.range_max # Random uniform distribution

            # Combine all four probabilities with their respective weights
            p_total = self.z_hit * p_hit + self.z_short * p_short + self.z_max * p_max + self.z_rand * p_rand
            
            # Use log-likelihood to avoid floating-point underflow
            if p_total > 1e-9: # Avoid log(0)
                total_log_likelihood += math.log(p_total)
            else:
                total_log_likelihood += math.log(1e-9) # Use a very small number

        # The final likelihood is the exponential of the sum of the log-likelihoods
        return math.exp(total_log_likelihood)


    def resample(self):
        """Resamples particles based on their weights (low variance resampling)."""
        new_particles = np.zeros_like(self.particles)
        
        # Low variance resampling
        r = uniform(0, 1.0 / self.num_particles)
        c = self.weights[0]
        i = 0
        
        for m in range(self.num_particles):
            U = r + m * (1.0 / self.num_particles)
            while U > c:
                i += 1
                c += self.weights[i]
            new_particles[m] = self.particles[i]

        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles


    def estimate_pose(self):
        """Calculates the estimated pose from the particle set."""
        # Use a weighted average
        self.estimated_pose[0] = np.average(self.particles[:, 0], weights=self.weights)
        self.estimated_pose[1] = np.average(self.particles[:, 1], weights=self.weights)
        
        # Average of angles needs to be handled carefully
        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 2]))
        self.estimated_pose[2] = math.atan2(sin_sum, cos_sum)

        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.map_frame
        pose_msg.pose.position.x = self.estimated_pose[0]
        pose_msg.pose.position.y = self.estimated_pose[1]
        pose_msg.pose.orientation = self.yaw_to_quaternion(self.estimated_pose[2])
        self.pose_pub.publish(pose_msg)

        # Publish map -> base_link transform for visualization
        self.publish_map_to_base_link_tf()


    def publish_map_to_base_link_tf(self):
        """Publishes the transform from map frame to base_link frame."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.base_frame
        
        t.transform.translation.x = self.estimated_pose[0]
        t.transform.translation.y = self.estimated_pose[1]
        t.transform.rotation = self.yaw_to_quaternion(self.estimated_pose[2])
        
        self.tf_broadcaster.sendTransform(t)


    def publish_particles(self):
        """Publishes the particle cloud for Rviz2 visualization."""
        particle_cloud_msg = PoseArray()
        particle_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        particle_cloud_msg.header.frame_id = self.map_frame
        
        poses = []
        for i in range(self.num_particles):
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.map_frame
            pose_stamped.pose.position.x = self.particles[i, 0]
            pose_stamped.pose.position.y = self.particles[i, 1]
            pose_stamped.pose.orientation = self.yaw_to_quaternion(self.particles[i, 2])
            poses.append(pose_stamped.pose)
            
        particle_cloud_msg.poses = poses
        self.particle_pub.publish(particle_cloud_msg)
        
    def quaternion_to_yaw(self, q):
        """Converts a quaternion to yaw angle."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def yaw_to_quaternion(self, yaw):
        """Converts a yaw angle to a quaternion."""
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q
        

def main(args=None):
    rclpy.init(args=args)
    mcl_node = MCLNode()
    try:
        rclpy.spin(mcl_node)
    except KeyboardInterrupt:
        pass
    finally:
        mcl_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
