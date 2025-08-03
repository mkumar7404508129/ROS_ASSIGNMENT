import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# =========================================================================================
#  PART 1: THE EKF SLAM "BRAIN" 🧠
# =========================================================================================
def normalize_angle(angle):
    """Wrap an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class EKF_SLAM:
    def __init__(self, initial_pose, motion_noise_std, meas_noise_std):
        # State Vector [x, y, theta, l1x, l1y, l2x, l2y, ...]'
        # The first 3 elements are the robot's pose. The rest are landmark positions.
        self.mu = np.array(initial_pose).reshape(3, 1)
        
        # Covariance Matrix: Represents the "blurriness" or uncertainty of our state.
        # Starts as 3x3 because we only know about the robot's pose initially.
        self.Sigma = np.zeros((3, 3))
        
        # Keep track of landmarks we've seen. Maps landmark ID to its position in the state vector.
        self.landmark_map = {}
        self.next_landmark_idx = 0
        
        # Noise parameters
        # Q_t: Measurement noise (for sensors)
        self.Q_t = np.diag([meas_noise_std['range']**2, meas_noise_std['bearing']**2])
        
        # R_t: Motion noise (for robot movement)
        self.R_t = np.diag([motion_noise_std['v']**2, motion_noise_std['omega']**2])

    def predict(self, u, dt):
        """
        The "Guess" step. Predicts the robot's next position and updates uncertainty.
        u: Control input [linear_velocity, angular_velocity]
        dt: Time step
        """
        v, omega = u
        theta = self.mu[2, 0]
        
        # How many landmarks are currently in our map?
        num_landmarks = self.next_landmark_idx
        N = 3 + 2 * num_landmarks # Total size of state vector
        
        # --- Update the State (The Guess) ---
        # The robot's pose is updated using a simple motion model.
        motion_update = np.array([
            [-v / omega * np.sin(theta) + v / omega * np.sin(theta + omega * dt)],
            [ v / omega * np.cos(theta) - v / omega * np.cos(theta - omega * dt)],
            [omega * dt]
        ])
        
        # A special matrix F helps us add this motion update to the full state vector.
        F = np.block([np.eye(3), np.zeros((3, 2 * num_landmarks))])
        self.mu = self.mu + F.T @ motion_update

        # --- Update the Covariance (The Blurriness Grows) ---
        # G is the Jacobian of the motion model. It tells us how a change in state affects the next state.
        G_robot = np.array([
            [0, 0, -v / omega * np.cos(theta) + v / omega * np.cos(theta + omega * dt)],
            [0, 0, -v / omega * np.sin(theta) + v / omega * np.sin(theta + omega * dt)],
            [0, 0, 0]
        ])
        G = np.eye(N) + F.T @ G_robot @ F

        # Update covariance: Sigma = G * Sigma * G^T + MotionNoise
        # V is the Jacobian of the motion model with respect to the control inputs.
        V = np.array([
            [(-np.sin(theta) + np.sin(theta + omega * dt)) / omega, 
             v * (np.sin(theta) - np.sin(theta + omega * dt)) / omega**2 + v * np.cos(theta + omega * dt) * dt / omega],
            [(np.cos(theta) - np.cos(theta + omega * dt)) / omega, 
             -v * (np.cos(theta) - np.cos(theta + omega * dt)) / omega**2 + v * np.sin(theta + omega * dt) * dt / omega],
            [0, dt]
        ])
        MotionNoise = F.T @ V @ self.R_t @ V.T @ F
        self.Sigma = G @ self.Sigma @ G.T + MotionNoise

    def update(self, measurements):
        """
        The "Check" step. Corrects the robot's pose and map using sensor measurements.
        measurements: A list of [landmark_id, range, bearing]
        """
        if not measurements:
            return

        rx, ry, rtheta = self.mu[0,0], self.mu[1,0], self.mu[2,0]
        
        for lm_id, z_range, z_bearing in measurements:
            # If we've never seen this landmark before, add it to our map.
            if lm_id not in self.landmark_map:
                # Add landmark ID to our map dictionary
                self.landmark_map[lm_id] = self.next_landmark_idx
                self.next_landmark_idx += 1
                
                # Extend the state vector `mu`
                lm_x = rx + z_range * np.cos(z_bearing + rtheta)
                lm_y = ry + z_range * np.sin(z_bearing + rtheta)
                self.mu = np.vstack([self.mu, [[lm_x], [lm_y]]])

                # Extend the covariance matrix `Sigma`
                old_size = self.Sigma.shape[0]
                self.Sigma = np.block([
                    [self.Sigma, np.zeros((old_size, 2))],
                    [np.zeros((2, old_size)), np.eye(2) * 1e6] # Initialize with large uncertainty
                ])
                continue # Done with this new landmark for now

            # If we HAVE seen this landmark before, perform the EKF update.
            lm_idx = self.landmark_map[lm_id]
            
            # Get landmark position from our current map belief
            lm_x = self.mu[3 + 2 * lm_idx, 0]
            lm_y = self.mu[3 + 2 * lm_idx + 1, 0]

            # --- Calculate Expected Measurement ---
            delta = np.array([lm_x - rx, lm_y - ry])
            q = delta.T @ delta
            expected_range = np.sqrt(q)
            expected_bearing = np.arctan2(delta[1], delta[0]) - rtheta
            
            z = np.array([[z_range], [z_bearing]])
            z_hat = np.array([[expected_range], [normalize_angle(expected_bearing)]])
            
            # --- Calculate Jacobian H ---
            # H maps a change in state to a change in measurement.
            Fj = np.zeros((5, self.mu.shape[0]))
            Fj[:3, :3] = np.eye(3)
            Fj[3:, 3 + 2 * lm_idx : 3 + 2 * lm_idx + 2] = np.eye(2)

            H_low = np.array([
                [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], 0, np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
                [delta[1], -delta[0], -q, -delta[1], delta[0]]
            ]) / q
            H = H_low @ Fj

            # --- Calculate Kalman Gain ---
            S = H @ self.Sigma @ H.T + self.Q_t
            K = self.Sigma @ H.T @ np.linalg.inv(S)

            # --- Update State and Covariance ---
            innovation = z - z_hat
            innovation[1] = normalize_angle(innovation[1]) # Handle angle wrapping
            self.mu = self.mu + K @ innovation
            self.mu[2] = normalize_angle(self.mu[2]) # Normalize robot's angle
            
            I = np.eye(self.mu.shape[0])
            self.Sigma = (I - K @ H) @ self.Sigma

# =========================================================================================
#  PART 2: THE PYBULLET SIMULATION 🤖
# =========================================================================================
def run_simulation(ekf, landmarks_true_pos, steps=500, dt=0.1):
    # --- Setup PyBullet ---
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    # --- Create Robot ---
    # We use a simple box as our robot
    robot_start_pos = [0, 0, 0.1]
    robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.1, 0.1])
    p.createMultiBody(baseCollisionShapeIndex=robotId, basePosition=robot_start_pos)
    
    # --- Create Landmarks ---
    # These are the "true" positions of objects in the world
    landmark_ids = []
    for i, pos in enumerate(landmarks_true_pos):
        lm_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.5)
        lm_body = p.createMultiBody(baseCollisionShapeIndex=lm_shape, basePosition=pos)
        landmark_ids.append(lm_body)
        p.changeVisualShape(lm_body, -1, rgbaColor=[0, 1, 0, 1]) # Green landmarks
        
    # --- History for Plotting ---
    history = {
        'true_path': [],
        'est_path': [],
        'true_landmarks': landmarks_true_pos,
        'est_landmarks': [],
        'final_sigma': None
    }
    
    # --- Simulation Loop ---
    for i in range(steps):
        # --- Define Robot's Action (e.g., move in a circle) ---
        v = 1.0  # constant linear velocity
        omega = 0.5  # constant angular velocity
        
        # --- Simulate Robot Motion with Noise ---
        v_noisy = v + np.random.randn() * ekf.R_t[0, 0]**0.5
        omega_noisy = omega + np.random.randn() * ekf.R_t[1, 1]**0.5
        
        pos, ori = p.getBasePositionAndOrientation(robotId)
        euler = p.getEulerFromQuaternion(ori)
        theta = euler[2]
        
        # Update robot position in simulation
        dx = v_noisy * dt * np.cos(theta)
        dy = v_noisy * dt * np.sin(theta)
        dtheta = omega_noisy * dt
        p.resetBasePositionAndOrientation(robotId, [pos[0]+dx, pos[1]+dy, pos[2]], p.getQuaternionFromEuler([0,0,theta+dtheta]))
        
        # Store ground truth path
        true_pos, true_ori = p.getBasePositionAndOrientation(robotId)
        true_euler = p.getEulerFromQuaternion(true_ori)
        history['true_path'].append([true_pos[0], true_pos[1], true_euler[2]])

        # --- EKF Prediction Step ---
        control_input = np.array([v, omega])
        ekf.predict(control_input, dt)
        
        # --- Simulate Sensor Measurements with Noise ---
        measurements = []
        robot_pos_true = np.array(history['true_path'][-1])
        
        for lm_id, lm_pos_true in enumerate(landmarks_true_pos):
            dist_x = lm_pos_true[0] - robot_pos_true[0]
            dist_y = lm_pos_true[1] - robot_pos_true[1]
            
            true_range = np.sqrt(dist_x**2 + dist_y**2)
            true_bearing = np.arctan2(dist_y, dist_x)
            
            # Add sensor noise
            noisy_range = true_range + np.random.randn() * ekf.Q_t[0,0]**0.5
            noisy_bearing = true_bearing - robot_pos_true[2] + np.random.randn() * ekf.Q_t[1,1]**0.5
            
            # We can only "see" landmarks within a certain range
            if noisy_range < 5.0:
                measurements.append([lm_id, noisy_range, normalize_angle(noisy_bearing)])

        # --- EKF Update Step ---
        ekf.update(measurements)
        
        # Store estimated path
        history['est_path'].append(ekf.mu[:3].flatten().tolist())
        
        time.sleep(dt)

    p.disconnect()
    
    # Store final map estimate
    for lm_id, lm_idx in ekf.landmark_map.items():
        history['est_landmarks'].append(ekf.mu[3+2*lm_idx:3+2*lm_idx+2].flatten().tolist())
    history['final_sigma'] = ekf.Sigma
    
    return history


# =========================================================================================
#  PART 3: THE PLOTTING FUNCTION 📊
# =========================================================================================
def plot_results(history):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract data from history
    true_path = np.array(history['true_path'])
    est_path = np.array(history['est_path'])
    true_landmarks = np.array(history['true_landmarks'])
    est_landmarks = np.array(history['est_landmarks'])
    sigma = history['final_sigma']

    # Plot paths
    ax.plot(true_path[:, 0], true_path[:, 1], 'b-', label='Real Robot Path')
    ax.plot(est_path[:, 0], est_path[:, 1], 'r--', label='Estimated Robot Path (EKF)')
    
    # Plot landmarks
    ax.scatter(true_landmarks[:, 0], true_landmarks[:, 1], s=100, c='g', marker='X', label='Real Landmarks')
    if est_landmarks.any():
        ax.scatter(est_landmarks[:, 0], est_landmarks[:, 1], s=100, c='r', marker='+', label='Estimated Landmarks (EKF)')
    
    # Plot uncertainty ellipses for landmarks
    for i in range(est_landmarks.shape[0]):
        # The covariance for the i-th landmark is a 2x2 sub-matrix
        lm_sigma = sigma[3+2*i : 3+2*i+2, 3+2*i : 3+2*i+2]
        
        # Calculate ellipse properties from covariance
        eigenvalues, eigenvectors = np.linalg.eigh(lm_sigma)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        # Use 95% confidence interval (2 * sqrt(eigenvalue))
        width, height = 2 * 2 * np.sqrt(eigenvalues) 

        ellipse = Ellipse(xy=est_landmarks[i], width=width, height=height, angle=angle,
                          edgecolor='r', fc='None', lw=1, ls='--')
        ax.add_patch(ellipse)

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("EKF SLAM Simulation Result")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    plt.show()


# =========================================================================================
#  MAIN SCRIPT EXECUTION
# =========================================================================================
if __name__ == '__main__':
    # --- Configuration ---
    # Define the true positions of landmarks in the world
    landmarks_true_positions = [
        [5, 5, 0.5],
        [3, -4, 0.5],
        [-6, -2, 0.5],
        [-4, 7, 0.5]
    ]

    # Define robot and sensor noise levels
    motion_noise = {'v': 0.1, 'omega': 0.05}  # Std dev for linear and angular velocity
    measurement_noise = {'range': 0.2, 'bearing': 0.1} # Std dev for range and bearing sensors

    # --- Initialize EKF ---
    # The robot starts at (0, 0) facing 0 degrees.
    initial_robot_pose = [0, 0, 0] 
    ekf = EKF_SLAM(initial_robot_pose, motion_noise, measurement_noise)

    # --- Run Simulation and Get History ---
    simulation_history = run_simulation(ekf, landmarks_true_positions)

    # --- Plot the Final Map and Paths ---
    plot_results(simulation_history)