import pybullet as p
import pybullet_data
import time
import numpy as np
import math

class PyBulletSimulator:
    """
    A standalone Python class to run a PyBullet simulation without ROS.
    It simulates a robot, its movement, and a laser scanner, now with
    a simple obstacle avoidance behavior.
    """
    def __init__(self):
        # Initialize PyBullet GUI
        # If p.DIRECT is used, the GUI will not be displayed
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        print('PyBullet Standalone Simulator started.')
        
        # Configure simulation time step
        self.sim_frequency_hz = 50.0
        self.time_step = 1.0 / self.sim_frequency_hz
        p.setTimeStep(self.time_step)
        
        # Load robot model to a corner to facilitate debugging of laser scans
        self.robot_id = p.loadURDF("r2d2.urdf", [8.0, 8.0, 0.1], useFixedBase=False)
        print(f"Robot loaded with ID: {self.robot_id}")

        # Load a simple environment (plane) and walls/obstacles
        self.plane_id = p.loadURDF("plane.urdf")
        self.wall_ids = self.create_simulation_environment()
        
        # --- Robot control parameters ---
        self.left_front_wheel_joint_index = 6
        self.left_rear_wheel_joint_index = 7
        self.right_front_wheel_joint_index = 2
        self.right_rear_wheel_joint_index = 3
        self.wheel_radius = 0.05
        self.wheel_base = 0.2
        
        # Log joint names for debugging
        print("\n--- Joint Information ---")
        for joint_id in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, joint_id)
            print(f"Joint {joint_id}: Name={joint_info[1].decode('utf-8')}, Type={joint_info[2]}")
        print("-------------------------\n")
        
        # --- Laser scan parameters ---
        self.num_laser_beams = 360
        self.laser_range_max = 10.0 # Increased max range for better avoidance
        self.laser_angle_min = -np.pi
        self.laser_angle_max = np.pi
        self.laser_angle_increment = (self.laser_angle_max - self.laser_angle_min) / self.num_laser_beams
        self.laser_height = 0.6
        self.debug_draw_laser = True

    def create_simulation_environment(self):
        """
        Creates the walls and obstacles in the PyBullet simulation.
        The values are hardcoded for this specific example but could be
        made configurable.
        """
        wall_ids = []
        wall_half_length = 10.0 
        wall_height = 0.5
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
        obs_height = 0.5
        obs_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[obs_half_size, obs_half_size, obs_height])
        obs_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[obs_half_size, obs_half_size, obs_height], rgbaColor=[0.2, 0.8, 0.2, 1])
        wall_ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obs_collision_shape, baseVisualShapeIndex=obs_visual_shape, basePosition=[0, 0, obs_height]))
        
        return wall_ids

    def apply_robot_control(self, linear_vel, angular_vel):
        """
        Applies velocity commands to the robot's wheels.
        Converts linear and angular velocities to individual wheel velocities.
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

    def simulate_and_get_laser_scan(self):
        """
        Simulates laser scan rays and returns a list of ranges.
        Uses p.rayTestBatch for efficiency.
        """
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot_id)
        _, _, robot_yaw = p.getEulerFromQuaternion(robot_ori)

        laser_origin_world = [robot_pos[0], robot_pos[1], robot_pos[2] + self.laser_height]

        ray_starts = []
        ray_ends = []
        for i in range(self.num_laser_beams):
            angle = self.laser_angle_min + i * self.laser_angle_increment
            world_angle = robot_yaw + angle
            ray_end = [
                laser_origin_world[0] + self.laser_range_max * math.cos(world_angle),
                laser_origin_world[1] + self.laser_range_max * math.sin(world_angle),
                laser_origin_world[2]
            ]
            ray_starts.append(laser_origin_world)
            ray_ends.append(ray_end)

        results = p.rayTestBatch(ray_starts, ray_ends)

        ranges = [float('inf')] * self.num_laser_beams
        hit_count = 0
        for i, result in enumerate(results):
            hit_fraction = result[0]
            # A hit fraction > 0.0 indicates a collision
            if hit_fraction > 0.0:
                distance = hit_fraction * self.laser_range_max
                ranges[i] = distance
                hit_count += 1
            
            # Draw debug lines based on hit or no-hit
            if self.debug_draw_laser:
                if hit_fraction > 0.0:
                    # Ray hit an object, use hit position from result
                    hit_position = result[3]
                    color = [0, 1, 0] # Green for a hit
                    p.addUserDebugLine(laser_origin_world, hit_position, lineColorRGB=color, lifeTime=self.time_step * 2)
                else:
                    # Ray did not hit an object, draw a red line to max range
                    color = [1, 0, 0] # Red for no hit
                    p.addUserDebugLine(laser_origin_world, ray_ends[i], lineColorRGB=color, lifeTime=self.time_step * 2)

        return ranges
        
    def run_obstacle_avoidance_loop(self):
        """
        Main simulation loop with a simple Braitenberg-like obstacle avoidance.
        The robot drives forward and turns away from obstacles detected by the laser.
        """
        print('Starting obstacle avoidance loop. Robot will drive and avoid walls.')
        
        # Obstacle avoidance parameters
        forward_speed = 1.0
        turn_speed = 0.5
        min_distance = 1.5 # The distance at which the robot starts reacting
        
        while True:
            # Get current laser scan data
            ranges = self.simulate_and_get_laser_scan()
            
            # Simple decision-making based on the forward-facing laser beams
            # Get the indices for the front-left and front-right sectors
            front_left_start_index = int(self.num_laser_beams * 0.1)
            front_left_end_index = int(self.num_laser_beams * 0.3)
            
            front_right_start_index = int(self.num_laser_beams * 0.7)
            front_right_end_index = int(self.num_laser_beams * 0.9)

            # Get the minimum range in each sector
            left_side_min_range = min(ranges[front_left_start_index:front_left_end_index])
            right_side_min_range = min(ranges[front_right_start_index:front_right_end_index])
            
            linear_vel = forward_speed
            angular_vel = 0.0
            
            # Avoidance logic
            if left_side_min_range < min_distance and right_side_min_range < min_distance:
                # Obstacle detected on both sides, turn around
                print(f"Obstacle close on both sides ({left_side_min_range:.2f}m, {right_side_min_range:.2f}m). Turning hard right.")
                angular_vel = -turn_speed * 2
            elif left_side_min_range < min_distance:
                # Obstacle on the left, turn right
                print(f"Obstacle close on left side ({left_side_min_range:.2f}m). Turning right.")
                angular_vel = -turn_speed
            elif right_side_min_range < min_distance:
                # Obstacle on the right, turn left
                print(f"Obstacle close on right side ({right_side_min_range:.2f}m). Turning left.")
                angular_vel = turn_speed
            else:
                # No obstacle, drive forward
                print("Path is clear. Driving forward.")
                angular_vel = 0.0
            
            self.apply_robot_control(linear_vel, angular_vel)
            
            # Step the simulation
            p.stepSimulation()
            
            # Pause to control simulation speed
            time.sleep(self.time_step)


def main():
    """
    Initializes the simulator and runs the obstacle avoidance loop.
    """
    # p.resetSimulation() can be useful if you are restarting the simulation
    # multiple times in the same PyBullet instance.
    sim = PyBulletSimulator()
    
    # Run the continuous obstacle avoidance loop
    try:
        sim.run_obstacle_avoidance_loop()
    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C to disconnect from PyBullet
        print("\nSimulation stopped by user.")
    finally:
        p.disconnect()

if __name__ == '__main__':
    main()
