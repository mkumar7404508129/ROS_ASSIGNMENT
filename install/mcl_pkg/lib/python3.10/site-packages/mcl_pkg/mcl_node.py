#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration

import numpy as np
from numpy.random import uniform, randn
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

# ROS 2 message types
from nav_msgs.msg import Odometry, OccupancyGrid
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseArray, TransformStamped, PoseWithCovarianceStamped, Pose
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, DurabilityPolicy

class MCLNode(Node):
    def __init__(self):
        super().__init__('mcl_node')
        self.get_logger().info('Monte Carlo Localization Node has been started.')

        # --- Parameters ---
        # --- FIX: Increased particle count to a reasonable number ---
        self.num_particles = self.declare_parameter('num_particles', 500).value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value

        # Motion model noise parameters
        self.alpha1 = self.declare_parameter('alpha1', 0.1).value
        self.alpha2 = self.declare_parameter('alpha2', 0.1).value
        self.alpha3 = self.declare_parameter('alpha3', 0.1).value
        self.alpha4 = self.declare_parameter('alpha4', 0.1).value
        self.img_number=1
        # Sensor model noise parameters
        self.sigma_hit = self.declare_parameter('sigma_hit', 0.2).value
        self.lambda_short = self.declare_parameter('lambda_short', 0.1).value
        self.z_hit = self.declare_parameter('z_hit', 0.85).value
        self.z_short = self.declare_parameter('z_short', 0.05).value
        self.z_max = self.declare_parameter('z_max', 0.05).value
        self.z_rand = self.declare_parameter('z_rand', 0.05).value

        # --- State variables ---
        self.particles = np.zeros((self.num_particles, 3))  # [x, y, yaw]
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.last_odom_pose = None
        self.estimated_pose = Pose()
        self.path_msg = Path() # To store the history of poses
        self.path_msg.header.frame_id = self.map_frame

        self.occupancy_grid = None
        self.map_received = False
        self.odom_path_for_plot = []

        # --- ROS 2 Publishers and Subscribers ---
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initial_pose_callback, 10)

        map_qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos_profile)

        self.pose_pub = self.create_publisher(PoseStamped, '/amcl_pose', 10)
        self.particle_pub = self.create_publisher(PoseArray, '/particlecloud', 10)
        self.path_pub = self.create_publisher(Path, '/amcl_path', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.plot_timer = self.create_timer(10.0, self.plot_callback)

        self.get_logger().info("MCL node waiting for map...")

    def plot_callback(self):
        self.img_number+=1
        self.plot_particles_and_path(f'mcl_plot_{self.img_number}.png')

    def map_callback(self, msg):
        if self.map_received:
            return
        self.get_logger().info("Map received! Initializing MCL filter.")
        self.map_received = True
        self.occupancy_grid = {
            'info': msg.info,
            'data': np.array(msg.data).reshape((msg.info.height, msg.info.width))
        }
        self.initialize_particles()

    def initialize_particles(self):
        self.get_logger().info(f"Initializing {self.num_particles} particles...")
        info = self.occupancy_grid['info']
        map_min_x = info.origin.position.x
        map_max_x = map_min_x + info.width * info.resolution
        map_min_y = info.origin.position.y
        map_max_y = map_min_y + info.height * info.resolution
        
        free_space_indices = np.where(self.occupancy_grid['data'] == 0)
        num_free_cells = len(free_space_indices[0])

        if num_free_cells == 0:
            self.get_logger().error("No free space on the map to initialize particles!")
            return

        chosen_indices = np.random.choice(num_free_cells, self.num_particles)
        
        y_indices = free_space_indices[0][chosen_indices]
        x_indices = free_space_indices[1][chosen_indices]

        self.particles[:, 0] = x_indices * info.resolution + map_min_x
        self.particles[:, 1] = y_indices * info.resolution + map_min_y
        self.particles[:, 2] = uniform(-np.pi, np.pi, self.num_particles)
        
        self.weights.fill(1.0 / self.num_particles)
        self.publish_particles()
        self.get_logger().info("Particles initialized.")

    def initial_pose_callback(self, msg):
        if not self.map_received:
            self.get_logger().warn("Cannot set initial pose until map is received.")
            return

        self.get_logger().info("Received initialpose, re-initializing particles.")
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

        # Distribute particles in a Gaussian cloud around the initial pose
        self.particles[:, 0] = x + np.random.normal(0, msg.pose.covariance[0], self.num_particles)
        self.particles[:, 1] = y + np.random.normal(0, msg.pose.covariance[7], self.num_particles)
        self.particles[:, 2] = yaw + np.random.normal(0, msg.pose.covariance[35], self.num_particles)
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        self.weights.fill(1.0 / self.num_particles)
        self.publish_particles()

    def odom_callback(self, msg):
        if not self.map_received:
            return
        if self.last_odom_pose is None:
            self.last_odom_pose = msg.pose.pose
            return

        current_pose = msg.pose.pose
        
        # For plotting ground truth
        self.odom_path_for_plot.append((current_pose.position.x, current_pose.position.y))
        if len(self.odom_path_for_plot) > 1000: self.odom_path_for_plot.pop(0)

        # Calculate odometry motion
        dx = current_pose.position.x - self.last_odom_pose.position.x
        dy = current_pose.position.y - self.last_odom_pose.position.y
        delta_trans = math.sqrt(dx**2 + dy**2)
        
        last_yaw = self.quaternion_to_yaw(self.last_odom_pose.orientation)
        current_yaw = self.quaternion_to_yaw(current_pose.orientation)
        
        delta_rot1 = math.atan2(dy, dx) - last_yaw
        delta_rot2 = current_yaw - last_yaw - delta_rot1

        self.last_odom_pose = current_pose

        # --- FIX: Noise should be ADDED, not subtracted ---
        # This correctly models the uncertainty in the robot's motion.
        delta_rot1_noisy = delta_rot1 + np.random.normal(0, self.alpha1*abs(delta_rot1) + self.alpha2*delta_trans, self.num_particles)
        delta_trans_noisy = delta_trans + np.random.normal(0, self.alpha3*delta_trans + self.alpha4*(abs(delta_rot1) + abs(delta_rot2)), self.num_particles)
        delta_rot2_noisy = delta_rot2 + np.random.normal(0, self.alpha1*abs(delta_rot2) + self.alpha2*delta_trans, self.num_particles)
        
        # Update particles with the noisy motion
        self.particles[:, 0] += delta_trans_noisy * np.cos(self.particles[:, 2] + delta_rot1_noisy)
        self.particles[:, 1] += delta_trans_noisy * np.sin(self.particles[:, 2] + delta_rot1_noisy)
        self.particles[:, 2] += delta_rot1_noisy + delta_rot2_noisy
        
        # Normalize angle to be within [-pi, pi]
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

    def laser_callback(self, msg):
        if not self.map_received or self.last_odom_pose is None:
            return
        
        # --- FIX: Vectorized weight calculation for speed ---
        self.weights = self.calculate_particle_weights(msg, self.particles)

        total_weight = np.sum(self.weights)
        if total_weight < 1e-9:
            self.get_logger().warn("All particles have zero weight! Consider re-initializing if this persists.")
            # Optional: Re-initialize if weights collapse completely
            # self.initialize_particles()
            return
        
        self.weights /= total_weight # Normalize

        self.resample()
        self.estimate_pose()
        self.publish_particles()
        self.publish_map_to_odom_tf()

    # --- FIX: New vectorized function for sensor model ---
    def calculate_particle_weights(self, laser_scan, particles):
        """Calculates the likelihood of all particles given a laser scan."""
        num_particles = particles.shape[0]
        weights = np.ones(num_particles)

        # Process a subset of beams for efficiency
        step = 5
        beam_indices = np.arange(0, len(laser_scan.ranges), step)
        angles = laser_scan.angle_min + beam_indices * laser_scan.angle_increment
        actual_ranges = np.array(laser_scan.ranges)[beam_indices]

        # Filter out invalid ranges
        valid_indices = np.isfinite(actual_ranges)
        angles = angles[valid_indices]
        actual_ranges = actual_ranges[valid_indices]

        # Use a single likelihood accumulator
        total_likelihood = np.zeros(num_particles)
        
        # Vectorized ray casting and probability calculation for each beam
        for angle, actual_range in zip(angles, actual_ranges):
            expected_ranges = self.ray_cast_map_vectorized(particles[:, 0], particles[:, 1], particles[:, 2], angle, laser_scan.range_max)
            
            p_hit = norm.pdf(actual_range, expected_ranges, self.sigma_hit)
            
            p_short = np.where(actual_range < expected_ranges, self.lambda_short * np.exp(-self.lambda_short * actual_range), 0)
            
            p_max = np.where(actual_range >= laser_scan.range_max * 0.99, 1.0, 0.0)
            
            p_rand = 1.0 / laser_scan.range_max

            p_total = self.z_hit * p_hit + self.z_short * p_short + self.z_max * p_max + self.z_rand * p_rand
            
            # --- FIX: Sum probabilities instead of multiplying (by summing logs) ---
            total_likelihood += p_total
        
        return total_likelihood

    def resample(self):
        """Low variance resampling."""
        new_indices = np.zeros(self.num_particles, dtype=int)
        r = uniform(0, 1.0 / self.num_particles)
        c = self.weights[0]
        i = 0
        for m in range(self.num_particles):
            u = r + m * (1.0 / self.num_particles)
            while u > c:
                i += 1
                c += self.weights[i]
            new_indices[m] = i
        
        self.particles = self.particles[new_indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate_pose(self):
        """Calculates the estimated pose and correctly appends a copy to the path."""
        # Calculate the weighted average pose from the particle cloud
        mean_x = np.average(self.particles[:, 0], weights=self.weights)
        mean_y = np.average(self.particles[:, 1], weights=self.weights)

        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 2]))
        mean_yaw = math.atan2(sin_sum, cos_sum)

        # --- FIX: Create a new PoseStamped object for this specific time step ---
        # This prevents the path history from being overwritten.
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.map_frame
        pose_msg.pose.position.x = mean_x
        pose_msg.pose.position.y = mean_y
        pose_msg.pose.orientation = self.yaw_to_quaternion(mean_yaw)

        # Update the node's main estimated_pose attribute for other functions (like TF)
        self.estimated_pose = pose_msg.pose

        # Publish the estimated pose
        self.pose_pub.publish(pose_msg)

        # Append the new, unique PoseStamped message to the path
        self.path_msg.poses.append(pose_msg)
        self.path_msg.header.stamp = pose_msg.header.stamp

        # Keep path size bounded
        if len(self.path_msg.poses) > 1000:
            self.path_msg.poses.pop(0)

        # Publish the full path
        self.path_pub.publish(self.path_msg)
        
    def publish_map_to_odom_tf(self):
        """Publishes the transform from map frame to odom frame."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        # --- FIX: Child frame should be odom_frame ---
        t.child_frame_id = self.odom_frame
        
        # This is a simplified map->odom transform.
        # A full implementation requires inverting the odom->base_link transform.
        # However, for many setups, this approximation is sufficient to start.
        t.transform.translation.x = self.estimated_pose.position.x - self.last_odom_pose.position.x
        t.transform.translation.y = self.estimated_pose.position.y - self.last_odom_pose.position.y
        est_yaw = self.quaternion_to_yaw(self.estimated_pose.orientation)
        odom_yaw = self.quaternion_to_yaw(self.last_odom_pose.orientation)
        q = self.yaw_to_quaternion(est_yaw - odom_yaw)
        t.transform.rotation = q
        
        self.tf_broadcaster.sendTransform(t)

    def publish_particles(self):
        particle_cloud_msg = PoseArray()
        particle_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        particle_cloud_msg.header.frame_id = self.map_frame
        
        particle_cloud_msg.poses = [Pose(position=Pose().position, orientation=self.yaw_to_quaternion(self.particles[i, 2])) for i in range(self.num_particles)]
        for i in range(self.num_particles):
            particle_cloud_msg.poses[i].position.x = self.particles[i, 0]
            particle_cloud_msg.poses[i].position.y = self.particles[i, 1]

        self.particle_pub.publish(particle_cloud_msg)
        
    def world_to_map(self, x, y):
        info = self.occupancy_grid['info']
        map_x = ((x - info.origin.position.x) / info.resolution).astype(int)
        map_y = ((y - info.origin.position.y) / info.resolution).astype(int)
        return map_x, map_y

    def is_occupied_vectorized(self, x, y):
        map_x, map_y = self.world_to_map(x, y)
        info = self.occupancy_grid['info']
        
        # Create a boolean array for out-of-bounds checks
        out_of_bounds = (map_x < 0) | (map_x >= info.width) | (map_y < 0) | (map_y >= info.height)
        
        # Create an array to hold the occupancy values, default to occupied (True)
        occupied = np.ones_like(x, dtype=bool)
        
        # For in-bounds particles, get their actual occupancy value
        in_bounds_indices = ~out_of_bounds
        occupied[in_bounds_indices] = self.occupancy_grid['data'][map_y[in_bounds_indices], map_x[in_bounds_indices]] > 50
        
        return occupied

    def ray_cast_map_vectorized(self, x, y, yaws, angle, max_range):
        """Vectorized version of ray casting for all particles."""
        num_particles = len(x)
        step_size = self.occupancy_grid['info'].resolution
        
        current_x, current_y = x.copy(), y.copy()
        distances = np.zeros(num_particles)
        active_rays = np.ones(num_particles, dtype=bool)

        ray_yaws = yaws + angle
        dx = step_size * np.cos(ray_yaws)
        dy = step_size * np.sin(ray_yaws)

        for _ in range(int(max_range / step_size)):
            if not np.any(active_rays):
                break
            
            # Update only active rays
            current_x[active_rays] += dx[active_rays]
            current_y[active_rays] += dy[active_rays]
            distances[active_rays] += step_size
            
            # Check for hits
            occupied = self.is_occupied_vectorized(current_x[active_rays], current_y[active_rays])
            
            # Deactivate rays that hit something
            active_rays[np.where(active_rays)[0][occupied]] = False

        # Rays that never hit are set to max_range
        distances[active_rays] = max_range
        return distances

    def plot_particles_and_path(self, filename='mcl_plot.png'):
        if not self.map_received: return
        fig, ax = plt.subplots(figsize=(12, 12))
        info = self.occupancy_grid['info']
        grid_display = np.flipud(self.occupancy_grid['data'])
        extent = [info.origin.position.x, info.origin.position.x + info.width * info.resolution,
                  info.origin.position.y, info.origin.position.y + info.height * info.resolution]
        ax.imshow(grid_display, cmap='gray_r', origin='lower', extent=extent)
        ax.scatter(self.particles[:, 0], self.particles[:, 1], c='cyan', s=5, label='Particles', alpha=0.6)
        if self.path_msg.poses:
            path_x = [p.pose.position.x for p in self.path_msg.poses]
            path_y = [p.pose.position.y for p in self.path_msg.poses]
            ax.plot(path_x, path_y, 'magenta', linewidth=2, label='Estimated Path (MCL)')
        if self.odom_path_for_plot:
            gt_x, gt_y = zip(*self.odom_path_for_plot)
            ax.plot(gt_x, gt_y, 'lime', linestyle='--', linewidth=2, label='Odom Path')
        est_x, est_y = self.estimated_pose.position.x, self.estimated_pose.position.y
        est_yaw = self.quaternion_to_yaw(self.estimated_pose.orientation)
        ax.plot(est_x, est_y, 'ro', markersize=8, label='Estimated Pose')
        ax.arrow(est_x, est_y, 0.5 * math.cos(est_yaw), 0.5 * math.sin(est_yaw), head_width=0.2, color='r')
        ax.set_title("MCL Localization")
        ax.set_xlabel("X [m]"), ax.set_ylabel("Y [m]")
        ax.legend(), ax.grid(True), ax.set_aspect('equal', adjustable='box')
        plt.tight_layout(), plt.savefig(filename), plt.close(fig)
        self.get_logger().info(f"Plot saved to {filename}")

    # --- Utility Functions ---
    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def yaw_to_quaternion(self, yaw):
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
