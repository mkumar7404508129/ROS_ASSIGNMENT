#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

class EKFSLAMNode(Node):
    """
    Implements a Feature-based SLAM algorithm using an Extended Kalman Filter.
    This node subscribes to Odometry and LaserScan data to build and maintain
    a map of landmarks while simultaneously localizing the robot.
    """
    def __init__(self):
        super().__init__('ekf_slam_node')
        self.get_logger().info('EKF SLAM Node has started.')

        # ROS 2 Subscribers
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10)
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_scan_callback,
            10)

        # ROS 2 Publishers
        self.pose_publisher = self.create_publisher(PoseStamped, '/slam/pose', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # EKF State and Covariance
        # State vector: [robot_x, robot_y, robot_yaw, landmark1_x, landmark1_y, ...]
        self.state_vector = np.zeros(3)  # Initial state: [x, y, yaw]
        # Initial covariance matrix for robot state
        self.covariance_matrix = np.eye(3) * 1e-9 
        self.last_odom_time = self.get_clock().now()
        
        # Odom and Sensor Noise (tunable parameters)
        # Process noise covariance for odometry
        self.odom_noise = np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2
        # Measurement noise covariance for a single landmark
        self.laser_noise = np.diag([0.1, np.deg2rad(1)]) ** 2

        # Map and Landmark Management
        # Dictionary to store landmark positions and their indices in the state vector
        # {landmark_id: {'pos_index': start_index, 'position': [x, y]}}
        self.landmarks = {}
        
        # A timer to publish the corrected pose and transform at a regular interval
        self.publisher_timer = self.create_timer(0.1, self.publish_slam_data)

        self.get_logger().info('Waiting for data...')

    def odometry_callback(self, msg):
        """
        Callback for new odometry data. Triggers the EKF prediction step.
        """
        current_time = self.get_clock().now()
        dt = (current_time - self.last_odom_time).nanoseconds / 1e9
        self.last_odom_time = current_time

        if dt < 0.01:
            return
        linear_x = msg.twist.twist.linear.x
        angular_z = msg.twist.twist.angular.z

        self.prediction_step(linear_x, angular_z, dt)

    def laser_scan_callback(self, msg):
        """
        Callback for new laser scan data. Triggers the EKF update step.
        """
        self.update_step(msg)

    def prediction_step(self, linear_vel, angular_vel, dt):
        """
        EKF Prediction Step: Predicts the robot's new state and covariance.
        """
        x, y, theta = self.state_vector[0], self.state_vector[1], self.state_vector[2]
        
        # Compute the change in robot pose based on a differential drive model
        if abs(angular_vel) > 1e-6:
            r = linear_vel / angular_vel
            d_x = -r * math.sin(theta) + r * math.sin(theta + angular_vel * dt)
            d_y = r * math.cos(theta) - r * math.cos(theta + angular_vel * dt)
        else:
            d_x = linear_vel * dt * math.cos(theta)
            d_y = linear_vel * dt * math.sin(theta)
        d_theta = angular_vel * dt
        
        # Predict the new robot state
        self.state_vector[0] += d_x
        self.state_vector[1] += d_y
        self.state_vector[2] += d_theta

        # Normalize yaw angle
        self.state_vector[2] = math.atan2(math.sin(self.state_vector[2]), math.cos(self.state_vector[2]))

        # --- Covariance Prediction ---
        # State transition Jacobian (F)
        # This relates the old state to the new state. It's an identity matrix for landmarks,
        # but has a non-zero element for the robot's motion.
        num_states = len(self.state_vector)
        F = np.eye(num_states)
        F[0, 2] = -d_y
        F[1, 2] = d_x

        # Motion noise Jacobian (G)
        # G is typically a 3x3 matrix relating motion noise to the change in robot pose
        G = np.zeros((3, 3))
        G[0, 0] = dt * math.cos(theta)
        G[1, 1] = dt * math.sin(theta)
        G[2, 2] = dt

        # Propagate covariance
        Q = np.zeros_like(self.covariance_matrix)
        Q[0:3, 0:3] = G @ self.odom_noise @ G.T
        self.covariance_matrix = F @ self.covariance_matrix @ F.T + Q

    def update_step(self, scan_msg):
        """
        EKF Update Step: Corrects the state and covariance using laser scan measurements.
        
        You MUST implement the following logic in this method:
        1.  Feature Extraction: Process `scan_msg` to find features (landmarks).
            - This is where you can integrate your traditional machine learning technique.
            - Example: Cluster laser points to identify potential landmarks.
            - The result should be a list of landmark positions relative to the robot.
        2.  Data Association: For each detected feature, determine if it's a new or known landmark.
            - Use a method like the Mahalanobis distance to compare new features with existing landmarks.
        3.  State and Covariance Update:
            - If a new landmark is found, use `self.augment_state()` to add it to the state vector and covariance matrix.
            - If a known landmark is re-observed, calculate the Kalman gain and perform the update.
        """
        # TODO: Implement feature extraction and data association here.
        
        # A simple placeholder for demonstration
        features = self.extract_features_from_scan(scan_msg)

        for feature in features:
            # TODO: Implement your data association logic
            is_new_landmark, associated_id = self.associate_data(feature)

            if is_new_landmark:
                # Transform the feature from robot frame to world frame
                x, y, yaw = self.state_vector[:3]
                feature_world_pos = self.transform_to_world_frame(feature, x, y, yaw)
                self.augment_state(feature_world_pos)
            else:
                self.kalman_update(feature, associated_id)

    def extract_features_from_scan(self, scan_msg):
        """
        TODO: Implement your feature extraction logic here.
        
        This is where you can apply a traditional machine learning technique.
        For example, a clustering algorithm (like K-means) or a simple classifier
        could be used to identify points that form a single object.
        
        Returns a list of (x, y) feature positions in the robot's local frame.
        """
        self.get_logger().info('Extracting features...')
        return []

    def associate_data(self, new_feature):
        """
        TODO: Implement your data association logic here.
        
        Use the Mahalanobis distance to match a new feature with an existing landmark.
        
        Returns:
            - is_new_landmark (bool): True if the feature is new, False otherwise.
            - associated_id (int): The ID of the associated landmark if found, None otherwise.
        """
        return True, None

    def augment_state(self, new_landmark_pos):
        """
        Augments the state vector and covariance matrix with a new landmark.
        """
        num_landmarks = len(self.landmarks)
        new_landmark_id = num_landmarks

        # Store landmark info
        self.landmarks[new_landmark_id] = {'pos_index': 3 + new_landmark_id * 2}

        # Augment the state vector
        self.state_vector = np.append(self.state_vector, new_landmark_pos)
        
        # Augment the covariance matrix
        num_states = len(self.state_vector)
        new_cov_matrix = np.zeros((num_states, num_states))
        old_size = self.covariance_matrix.shape[0]
        
        new_cov_matrix[:old_size, :old_size] = self.covariance_matrix
        
        # TODO: Fill in the remaining blocks of the new covariance matrix.
        # This involves calculating the Jacobian of the measurement model and using
        # the current robot pose and covariance.
        
        self.covariance_matrix = new_cov_matrix
        self.get_logger().info(f'New landmark added. Total landmarks: {len(self.landmarks)}')

    def kalman_update(self, measurement, landmark_id):
        """
        TODO: Implement the Kalman update step for an associated landmark.
        
        1. Calculate the expected measurement `h(x_k)`
        2. Compute the measurement Jacobian `H`
        3. Calculate the Kalman gain `K`
        4. Update the state vector and covariance matrix
        """
        pass

    def publish_slam_data(self):
        """
        Publishes the corrected robot pose and the transform from map to odom.
        """
        if len(self.state_vector) < 3:
            return

        # Publish the corrected robot pose in the 'map' frame
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = self.state_vector[0]
        pose_msg.pose.position.y = self.state_vector[1]
        
        quat = self.get_quaternion_from_yaw(self.state_vector[2])
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self.pose_publisher.publish(pose_msg)
        
        # Publish the transform from 'map' to 'base_link'
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.state_vector[0]
        t.transform.translation.y = self.state_vector[1]
        t.transform.translation.z = 0.0
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(t)

    def get_quaternion_from_yaw(self, yaw):
        """Helper to convert yaw angle to a quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(0)
        sp = math.sin(0)
        cr = math.cos(0)
        sr = math.sin(0)
        
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy
        
        return [x, y, z, w]
        
    def transform_to_world_frame(self, pos_robot_frame, x_robot, y_robot, yaw_robot):
        """Helper function to transform a point from robot to world frame."""
        px_robot, py_robot = pos_robot_frame
        
        cos_yaw = math.cos(yaw_robot)
        sin_yaw = math.sin(yaw_robot)

        px_world = x_robot + px_robot * cos_yaw - py_robot * sin_yaw
        py_world = y_robot + px_robot * sin_yaw + py_robot * cos_yaw

        return np.array([px_world, py_world])

def main(args=None):
    rclpy.init(args=args)
    node = EKFSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()