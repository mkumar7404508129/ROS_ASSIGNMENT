#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import os
import math

# --- NEW: Import messages for RViz2 visualization ---
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
# ---

# ROS 2 message types
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# =========================================================================================
#  PART 1: THE EKF SLAM "BRAIN" (Unchanged)
# =========================================================================================
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class EKF_SLAM:
    def __init__(self, initial_pose, motion_noise_std, meas_noise_std):
        self.mu = np.array(initial_pose).reshape(3, 1)
        self.Sigma = np.zeros((3, 3))
        self.landmark_map = {}
        self.next_landmark_idx = 0
        self.Q_t = np.diag([meas_noise_std['range']**2, meas_noise_std['bearing']**2])
        self.R_t_from_odom = np.diag([motion_noise_std['v']**2, motion_noise_std['omega']**2])

    def predict(self, v, omega, dt):
        theta = self.mu[2, 0]
        num_landmarks = self.next_landmark_idx
        N = 3 + 2 * num_landmarks
        motion_update = np.array([[v*dt*np.cos(theta)], [v*dt*np.sin(theta)], [omega*dt]])
        F = np.block([np.eye(3), np.zeros((3, 2 * num_landmarks))])
        self.mu = self.mu + F.T @ motion_update
        self.mu[2] = normalize_angle(self.mu[2])
        G_robot = np.array([[0,0,-v*dt*np.sin(theta)], [0,0,v*dt*np.cos(theta)], [0,0,0]])
        G = np.eye(N) + F.T @ G_robot @ F
        R_motion = np.diag([0.1**2, 0.1**2, np.deg2rad(1.0)**2])
        MotionNoise = F.T @ R_motion @ F
        self.Sigma = G @ self.Sigma @ G.T + MotionNoise

    def update(self, measurements):
        if not measurements: return
        rx, ry, rtheta = self.mu[0,0], self.mu[1,0], self.mu[2,0]
        for lm_id, z_range, z_bearing in measurements:
            if lm_id not in self.landmark_map:
                self.landmark_map[lm_id] = self.next_landmark_idx
                self.next_landmark_idx += 1
                lm_x = rx + z_range * np.cos(z_bearing + rtheta)
                lm_y = ry + z_range * np.sin(z_bearing + rtheta)
                self.mu = np.vstack([self.mu, [[lm_x], [lm_y]]])
                old_size = self.Sigma.shape[0]
                self.Sigma = np.block([
                    [self.Sigma, np.zeros((old_size, 2))],
                    [np.zeros((2, old_size)), np.eye(2) * 1e3]
                ])
                continue
            lm_idx = self.landmark_map[lm_id]
            lm_x, lm_y = self.mu[3+2*lm_idx, 0], self.mu[3+2*lm_idx+1, 0]
            delta = np.array([lm_x-rx, lm_y-ry])
            q = delta.T @ delta
            expected_range = np.sqrt(q)
            expected_bearing = np.arctan2(delta[1], delta[0]) - rtheta
            z = np.array([[z_range], [z_bearing]])
            z_hat = np.array([[expected_range], [normalize_angle(expected_bearing)]])
            Fj = np.zeros((5, self.mu.shape[0]))
            Fj[:3,:3] = np.eye(3)
            Fj[3:, 3+2*lm_idx:3+2*lm_idx+2] = np.eye(2)
            H_low = np.array([[-np.sqrt(q)*delta[0],-np.sqrt(q)*delta[1],0,np.sqrt(q)*delta[0],np.sqrt(q)*delta[1]], [delta[1],-delta[0],-q,-delta[1],delta[0]]]) / q
            H = H_low @ Fj
            S = H @ self.Sigma @ H.T + self.Q_t
            K = self.Sigma @ H.T @ np.linalg.inv(S)
            innovation = z - z_hat
            innovation[1] = normalize_angle(innovation[1])
            self.mu += K @ innovation
            self.mu[2] = normalize_angle(self.mu[2])
            self.Sigma = (np.eye(self.mu.shape[0]) - K @ H) @ self.Sigma

# =========================================================================================
#  PART 2: THE CORRECTED ROS 2 NODE WITH RViz2 PUBLISHERS
# =========================================================================================
class EkfSlamNode(Node):
    def __init__(self):
        super().__init__('ekf_slam_node')
        self.get_logger().info("EKF SLAM Node with RViz2 Publishers Started.")
        
        initial_pose = [4.0, 4.0, 0.0]
        motion_noise = {'v': 0.1, 'omega': 0.1}
        measurement_noise = {'range': 0.2, 'bearing': 0.1}
        self.ekf = EKF_SLAM(initial_pose, motion_noise, measurement_noise)

        # --- NEW: Publishers for RViz2 ---
        self.pose_publisher = self.create_publisher(PoseWithCovarianceStamped, '/ekf_pose', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/ekf_landmarks', 10)
        # ---
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.last_odom_time = None
        self.last_pose = np.array(initial_pose)
        self.history = {'true_path': [], 'est_path': []}
        
        self.step_counter = 0
        self.plot_update_freq = 100  # << MODIFIED: Less frequent updates
        self.output_dir = "slam_output_corrected"
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(f"A single plot file will be updated in '{self.output_dir}/'")
        
        self.next_landmark_id = 0
        self.association_threshold = 1.0

    def odom_callback(self, msg):
        current_time = self.get_clock().now()
        if self.last_odom_time is None: self.last_odom_time = current_time; return
        dt = (current_time - self.last_odom_time).nanoseconds / 1e9
        if dt == 0: return
        self.step_counter += 1
        pos, ori = msg.pose.pose.position, msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        current_pose = np.array([pos.x, pos.y, yaw])
        dx, dy = current_pose[0]-self.last_pose[0], current_pose[1]-self.last_pose[1]
        dtheta = normalize_angle(current_pose[2] - self.last_pose[2])
        v, omega = np.sqrt(dx**2 + dy**2) / dt, dtheta / dt
        self.ekf.predict(v, omega, dt)
        self.last_odom_time, self.last_pose = current_time, current_pose
        self.history['true_path'].append(current_pose)
        self.history['est_path'].append(self.ekf.mu[:3].flatten().tolist())
        if self.step_counter % self.plot_update_freq == 0:
            self.get_logger().info(f"Step {self.step_counter}: Updating plot and RViz markers.")
            self.update_and_save_plot()
            self.publish_rviz_visualizations() # << NEW

    def scan_callback(self, msg):
        detected_features = self.detect_features_from_scan(msg)
        measurements = self.associate_measurements(detected_features)
        if measurements: self.ekf.update(measurements)

    def associate_measurements(self, detected_features):
        measurements = []
        rx, ry, rtheta = self.ekf.mu[:3, 0]
        known_landmarks_map = {idx: id for id, idx in self.ekf.landmark_map.items()}
        for r, b in detected_features:
            detected_x, detected_y = rx + r * np.cos(b + rtheta), ry + r * np.sin(b + rtheta)
            best_dist, best_match_id = float('inf'), -1
            if self.ekf.mu.shape[0] > 3:
                landmarks_xy = self.ekf.mu[3:].reshape(-1, 2)
                distances = np.linalg.norm(landmarks_xy - np.array([detected_x, detected_y]), axis=1)
                if np.min(distances) < self.association_threshold:
                    best_dist = np.min(distances)
                    best_match_id = known_landmarks_map[np.argmin(distances)]
            if best_dist < self.association_threshold:
                measurements.append([best_match_id, r, b])
            else:
                lm_id = self.next_landmark_id
                self.next_landmark_id += 1
                measurements.append([lm_id, r, b])
        return measurements
    
    def detect_features_from_scan(self, msg):
        features, ranges, cluster = [], np.array(msg.ranges), []
        for i in range(len(ranges)):
            if msg.range_min < ranges[i] < msg.range_max:
                if not cluster or abs(ranges[i]-ranges[i-1]) < 0.2: cluster.append(i)
                else:
                    if 2 < len(cluster) < 50: features.append(self.process_cluster(cluster, msg))
                    cluster = [i]
            elif cluster:
                if 2 < len(cluster) < 50: features.append(self.process_cluster(cluster, msg))
                cluster = []
        if cluster and 2 < len(cluster) < 50: features.append(self.process_cluster(cluster, msg))
        return [f for f in features if f is not None]

    def process_cluster(self, cluster_indices, msg):
        avg_range = np.mean([msg.ranges[i] for i in cluster_indices])
        avg_index = int(np.mean(cluster_indices))
        avg_bearing = msg.angle_min + avg_index * msg.angle_increment
        return (avg_range, normalize_angle(avg_bearing))

    # --- NEW: Function to publish all RViz2 info ---
    def publish_rviz_visualizations(self):
        now = self.get_clock().now().to_msg()
        
        # Publish estimated pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y = self.ekf.mu[0,0], self.ekf.mu[1,0]
        qx, qy, qz, qw = quaternion_from_euler(0, 0, self.ekf.mu[2,0])
        pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w = qx, qy, qz, qw
        
        # Map 2D covariance to 6D
        cov = np.zeros((6, 6))
        cov[0,0] = self.ekf.Sigma[0,0] # x-x
        cov[0,1] = self.ekf.Sigma[0,1] # x-y
        cov[1,0] = self.ekf.Sigma[1,0] # y-x
        cov[1,1] = self.ekf.Sigma[1,1] # y-y
        cov[5,5] = self.ekf.Sigma[2,2] # yaw-yaw
        pose_msg.pose.covariance = cov.flatten().tolist()
        self.pose_publisher.publish(pose_msg)

        # Publish landmark markers
        marker_array = MarkerArray()
        # Delete all previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for lm_id, lm_idx in self.ekf.landmark_map.items():
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = 'odom'
            marker.ns = "ekf_landmarks"
            marker.id = lm_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y = self.ekf.mu[3+2*lm_idx,0], self.ekf.mu[3+2*lm_idx+1,0]
            marker.scale.x, marker.scale.y, marker.scale.z = 0.3, 0.3, 0.3
            marker.color.a, marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0, 1.0 # Magenta
            marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)

    def update_and_save_plot(self):
        # << MODIFIED: Overwrite a single file now >>
        filename = os.path.join(self.output_dir, "slam_latest_state.png")
        fig, ax = plt.subplots(figsize=(12, 12))
        plot_true_map(ax)
        ax.plot(np.array(self.history['true_path'])[:,0], np.array(self.history['true_path'])[:,1], 'b-', lw=2, label='Real Robot Path')
        ax.plot(np.array(self.history['est_path'])[:,0], np.array(self.history['est_path'])[:,1], 'r--', lw=2, label='Estimated Robot Path (EKF)')
        if self.ekf.mu.shape[0] > 3:
            est_landmarks = self.ekf.mu[3:].reshape(-1, 2)
            ax.scatter(est_landmarks[:, 0], est_landmarks[:, 1], s=120, c='m', marker='P', label='Estimated Landmarks')
            for i in range(len(self.ekf.landmark_map)):
                lm_sigma = self.ekf.Sigma[3+2*i:3+2*i+2, 3+2*i:3+2*i+2]
                eigenvalues, eigenvectors = np.linalg.eigh(lm_sigma)
                angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
                width, height = 2 * 3 * np.sqrt(np.abs(eigenvalues))
                ellipse = Ellipse(xy=est_landmarks[i], width=width, height=height, angle=angle, edgecolor='m', fc='None', lw=1.5, ls='--')
                ax.add_patch(ellipse)
        ax.set_xlabel("X (m)"), ax.set_ylabel("Y (m)"), ax.set_title(f"EKF SLAM State @ Step {self.step_counter}"), ax.legend()
        ax.grid(True), ax.set_aspect('equal', 'box'), ax.set_xlim(-12, 12), ax.set_ylim(-12, 12)
        plt.savefig(filename), plt.close(fig)

def plot_true_map(ax):
    ax.add_patch(Rectangle((-10,-10),20,0.2,color='k')), ax.add_patch(Rectangle((-10,9.8),20,0.2,color='k'))
    ax.add_patch(Rectangle((-10,-10),0.2,20,color='k')), ax.add_patch(Rectangle((9.8,-10),0.2,20,color='k'))
    ax.add_patch(Rectangle((-1,-1),2,2,color='k')), ax.add_patch(plt.Circle((-6,0),0.5,color='k'))
    ax.add_patch(plt.Circle((6,0),0.5,color='k')), ax.add_patch(plt.Circle((0,6),0.5,color='k'))
    ax.add_patch(plt.Circle((0,-6),0.5,color='k')), ax.add_patch(Rectangle((5.5,5.5),1,1,color='k'))
    ax.add_patch(Rectangle((-6.5,5.5),1,1,color='k')), ax.add_patch(Rectangle((5.5,-6.5),1,1,color='k'))
    ax.add_patch(Rectangle((-6.5,-6.5),1,1,color='k'))

def euler_from_quaternion(q_list):
    x, y, z, w = q_list
    t0,t1,t2 = +2.0*(w*x+y*z),+1.0-2.0*(x*x+y*y),+2.0*(w*y-z*x)
    t2 = +1.0 if t2 > +1.0 else (-1.0 if t2 < -1.0 else t2)
    t3,t4 = +2.0*(w*z+x*y),+1.0-2.0*(y*y+z*z)
    return np.arctan2(t0,t1), np.arcsin(t2), np.arctan2(t3,t4)

# --- NEW: Helper function to convert yaw to quaternion ---
def quaternion_from_euler(roll, pitch, yaw):
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return x, y, z, w

def main(args=None):
    rclpy.init(args=args)
    node = EkfSlamNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.get_logger().info("Shutting down EKF SLAM node.")
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()