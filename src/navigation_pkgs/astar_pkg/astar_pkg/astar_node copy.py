import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import heapq
import math

# --- A* Algorithm for 3D Continuous Space ---

class Node:
    """A node class for A* Pathfinding in continuous space"""
    def __init__(self, position, parent=None):
        self.parent = parent
        self.position = np.array(position)  # (x, y, yaw)

        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost from current node to end
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        # Compare nodes based on a discretized position to avoid floating point issues
        return np.linalg.norm(self.position[:2] - other.position[:2]) < 0.1

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        # Hash based on a rounded position to group close nodes
        return hash(tuple(np.round(self.position[:2], 1)))

def heuristic(a_pos, b_pos):
    """Calculate the Euclidean distance heuristic for 2D plane"""
    return np.linalg.norm(a_pos[:2] - b_pos[:2])

def is_collision(robot_id, obstacles):
    """Check if the robot is in collision with any of the obstacles."""
    for obs_id in obstacles:
        closest_points = p.getClosestPoints(robot_id, obs_id, distance=0.0)
        if len(closest_points) > 0:
            return True # Collision detected
    return False

def a_star_3d(robot_id, obstacles, start_pos, end_pos):
    """
    Returns a path of (x, y, yaw) tuples in a continuous 3D space.
    """
    start_node = Node(start_pos)
    end_node = Node(end_pos)

    open_list = []
    closed_set = set()

    heapq.heappush(open_list, start_node)
    explored_points = []

    # Define possible moves: (forward_dist, turn_angle_in_radians)
    # A smaller set of moves makes the search faster
    moves = [(1.0, 0), (0.5, np.deg2rad(45)), (0.5, np.deg2rad(-45))]

    while open_list:
        current_node = heapq.heappop(open_list)
        
        if current_node in closed_set:
            continue
        
        closed_set.add(current_node)
        explored_points.append(current_node.position)
        
        # Check if we reached the goal
        if heuristic(current_node.position, end_node.position) < 1.0:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1], explored_points

        # Generate children nodes from moves
        for move_dist, move_angle in moves:
            # New position and orientation (yaw)
            new_yaw = current_node.position[2] + move_angle
            new_x = current_node.position[0] + move_dist * np.cos(new_yaw)
            new_y = current_node.position[1] + move_dist * np.sin(new_yaw)
            node_position = (new_x, new_y, new_yaw)
            
            # --- Collision Check ---
            # Temporarily move robot to the new position to check for collisions
            p.resetBasePositionAndOrientation(robot_id, [new_x, new_y, 0.2], p.getQuaternionFromEuler([0, 0, new_yaw]))
            if is_collision(robot_id, obstacles):
                continue # Path is blocked

            new_node = Node(node_position, current_node)
            new_node.g = current_node.g + move_dist # Cost is the distance moved
            new_node.h = heuristic(new_node.position, end_node.position)
            new_node.f = new_node.g + new_node.h
            
            # Check if this node is already in the open list with a lower cost
            if any(open_node for open_node in open_list if new_node == open_node and new_node.g >= open_node.g):
                continue
            
            heapq.heappush(open_list, new_node)
            
    # Reset robot to its start pos if path fails
    p.resetBasePositionAndOrientation(robot_id, [start_pos[0], start_pos[1], 0.2], p.getQuaternionFromEuler([0, 0, start_pos[2]]))
    return None, explored_points

# --- Visualization Functions ---
def setup_pybullet_env():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[5, 5, 0])
    
    # Load robot
    start_orientation = p.getQuaternionFromEuler([0,0,0])
    robot_id = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.2], start_orientation)

    # Create 3D obstacles
    obstacles = []
    # Cube obstacle
    cube_id = p.loadURDF("cube.urdf", [5, 5, 0.5], globalScaling=2)
    obstacles.append(cube_id)
    # Sphere obstacle
    sphere_id = p.loadURDF("sphere_small.urdf", [9, 2, 0.5], globalScaling=5)
    obstacles.append(sphere_id)
    # Another cube
    cube_id2 = p.loadURDF("cube.urdf", [2, 9, 0.5], globalScaling=2.5)
    obstacles.append(cube_id2)

    return robot_id, obstacles

def draw_path(path):
    """Draws a line in PyBullet to show the path."""
    for i in range(len(path) - 1):
        p.addUserDebugLine(
            [path[i][0], path[i][1], 0.1],
            [path[i+1][0], path[i+1][1], 0.1],
            lineColorRGB=[0, 0, 1],
            lineWidth=5
        )

def animate_robot_path(robot_id, path):
    """Animates the robot model along the calculated path."""
    print("Animating path...")
    for pos in path:
        p.resetBasePositionAndOrientation(robot_id, [pos[0], pos[1], 0.2], p.getQuaternionFromEuler([0, 0, pos[2]]))
        time.sleep(0.05) # Slow down animation

def plot_3d_astar_graph(start, end, path, explored_points, obstacles_info):
    """Plots a top-down 2D view of the 3D search."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot explored points
    if explored_points:
        explored_x = [p[0] for p in explored_points]
        explored_y = [p[1] for p in explored_points]
        ax.scatter(explored_x, explored_y, c='yellow', alpha=0.3, label='Explored Points', s=50)

    # Plot obstacles
    for obs in obstacles_info:
        if obs['type'] == 'cube':
            ax.add_patch(plt.Rectangle((obs['pos'][0]-obs['scale'], obs['pos'][1]-obs['scale']), obs['scale']*2, obs['scale']*2, color='gray', label='Obstacle'))
        elif obs['type'] == 'sphere':
            ax.add_patch(plt.Circle((obs['pos'][0], obs['pos'][1]), obs['scale']*0.5, color='gray'))

    # Plot final path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, color='blue', linewidth=3, marker='o', markersize=5, label='Final Path')
    
    # Mark Start and End
    ax.plot(start[0], start[1], 'go', markersize=15, label='Start')
    ax.plot(end[0], end[1], 'ro', markersize=15, label='End')

    ax.set_title("A* Algorithm Top-Down View", fontsize=16)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    robot_id, obstacles = setup_pybullet_env()
    
    # Define start and end (x, y, yaw)
    start_pos = (0, 0, np.deg2rad(45))
    end_pos = (12, 12, 0)

    # Set robot to start position
    p.resetBasePositionAndOrientation(robot_id, [start_pos[0], start_pos[1], 0.2], p.getQuaternionFromEuler([0, 0, start_pos[2]]))

    print("Running A* search in 3D space...")
    path, explored = a_star_3d(robot_id, obstacles, start_pos, end_pos)
    
    if path:
        print("Path found!")
        draw_path(path)
        animate_robot_path(robot_id, path)
        
        # Get obstacle info for plotting
        obstacles_info_for_plot = [
            {'type': 'cube', 'pos': [5, 5], 'scale': 1.0}, # halfExtents for cube is 1, so scale is 1
            {'type': 'sphere', 'pos': [9, 2], 'scale': 2.5}, # radius for sphere_small is 0.25, so 0.25*5
            {'type': 'cube', 'pos': [2, 9], 'scale': 1.25} # halfExtents * scale
        ]
        
        print("Generating Matplotlib graph...")
        plot_3d_astar_graph(start_pos, end_pos, path, explored, obstacles_info_for_plot)
    else:
        print("No path found.")

    print("Simulation finished. Close the window to exit.")
    # Keep simulation open to view the result
    while p.isConnected():
        try:
            time.sleep(1)
        except p.error:
            break