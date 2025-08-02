import pybullet as p
import pybullet_data
import time
import numpy as np

# --- Global variable to store debug line IDs ---
debug_lines = []

# --- Helper Function for Robot Motion ---
def set_velocity(robot_id, linear_vel, angular_vel):
    left_wheel_joint_index = 2
    right_wheel_joint_index = 3
    wheel_base = 0.6
    wheel_radius = 0.1
    
    v_left = (2 * linear_vel - angular_vel * wheel_base) / (2 * wheel_radius)
    v_right = (2 * linear_vel + angular_vel * wheel_base) / (2 * wheel_radius)

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=left_wheel_joint_index,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=v_left
    )
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=right_wheel_joint_index,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=v_right
    )

# --- MODIFIED Helper Function for Laser Scan Simulation ---
def get_laser_scan(robot_id, num_rays=360, max_range=5.0, draw=False):
    """
    Simulates a 2D laser scan and optionally draws debug lines.
    """
    global debug_lines
    robot_pos, robot_orn_quat = p.getBasePositionAndOrientation(robot_id)
    _, _, robot_yaw = p.getEulerFromQuaternion(robot_orn_quat)

    ray_from = []
    ray_to = []
    laser_z_offset = 0.2
    
    for i in range(num_rays):
        angle = robot_yaw + (2 * np.pi * i / num_rays)
        from_pos = [robot_pos[0], robot_pos[1], robot_pos[2] + laser_z_offset]
        to_pos = [
            from_pos[0] + max_range * np.cos(angle),
            from_pos[1] + max_range * np.sin(angle),
            from_pos[2]
        ]
        ray_from.append(from_pos)
        ray_to.append(to_pos)

    results = p.rayTestBatch(rayFromPositions=ray_from, rayToPositions=ray_to)

    # If drawing, remove the previous lines
    if draw:
        for line_id in debug_lines:
            p.removeUserDebugItem(line_id)
        debug_lines.clear()

    ranges = []
    for i in range(num_rays):
        hit_fraction = results[i][2]
        hit_pos = results[i][3]
        
        if hit_fraction == 1.0:
            # No hit
            ranges.append(max_range)
            if draw:
                # Draw a green line for a "miss"
                line_id = p.addUserDebugLine(ray_from[i], ray_to[i], [0, 1, 0]) # Green
                debug_lines.append(line_id)
        else:
            # Hit object
            dist = np.linalg.norm(np.array(hit_pos) - np.array(ray_from[i]))
            ranges.append(dist)
            if draw:
                # Draw a red line for a "hit"
                line_id = p.addUserDebugLine(ray_from[i], hit_pos, [1, 0, 0]) # Red
                debug_lines.append(line_id)
            
    return ranges

# --- Main Simulation Setup ---
if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    planeId = p.loadURDF("plane.urdf")

    wall_shapes = [
        [[5, 0.1, 1], [0, 5, 0.5], [0,0,0,1]], [[5, 0.1, 1], [0, -5, 0.5], [0,0,0,1]],
        [[0.1, 5, 1], [5, 0, 0.5], [0,0,0,1]], [[0.1, 5, 1], [-5, 0, 0.5], [0,0,0,1]]
    ]
    for shape in wall_shapes:
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=shape[0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=shape[1], baseOrientation=shape[2])

    start_pos = [0, 0, 0.1]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF("r2d2.urdf", start_pos, start_orientation)
    
    print("Simulation starting... Close the window to exit.")
    try:
        while True:
            set_velocity(robot_id, linear_vel=0.5, angular_vel=0.8)

            # Call the sensor simulation function and tell it to draw the rays
            laser_ranges = get_laser_scan(robot_id, num_rays=90, draw=False)

            p.stepSimulation()
            time.sleep(1./240.)

    except p.error as e:
        print(f"PyBullet error: {e}")
    finally:
        p.disconnect()
        print("Simulation finished and disconnected.")