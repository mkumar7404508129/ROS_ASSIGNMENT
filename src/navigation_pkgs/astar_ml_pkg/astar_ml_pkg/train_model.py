#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy

import numpy as np
import heapq
import math
import time
import random
import pickle
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # <<< NEW: Import tqdm for the progress bar

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- A* Node (Unchanged) ---
class AStarNode:
    def __init__(self, parent=None, position=None):
        self.parent, self.position, self.g, self.h, self.f = parent, position, 0, 0, 0
    def __eq__(self, other): return self.position == other.position
    def __lt__(self, other): return self.f < other.f
    def __hash__(self): return hash(self.position)

# --- WORKER FUNCTION for multiprocessing ---
def run_a_star_worker(args):
    """
    A self-contained worker that runs a single A* search.
    """
    grid, start, end = args
    # A* search logic is now inside the worker
    start_node, end_node = AStarNode(None, start), AStarNode(None, end)
    open_list, closed_set = [], set()
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            # <<< MODIFIED: Return start/end points along with the path
            return (start, end, path[::-1])

        for new_position in [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if not (0 <= node_position[0] < len(grid) and 0 <= node_position[1] < len(grid[0])) \
               or grid[node_position[0]][node_position[1]] > 50 or node_position in closed_set:
                continue

            new_node = AStarNode(current_node, node_position)
            new_node.g = current_node.g + math.sqrt(new_position[0]**2 + new_position[1]**2)
            new_node.h = abs(node_position[0] - end_node.position[0]) + abs(node_position[1] - end_node.position[1])
            new_node.f = new_node.g + new_node.h

            if any(open_node for open_node in open_list if new_node == open_node and new_node.g >= open_node.g):
                continue

            heapq.heappush(open_list, new_node)

    # <<< MODIFIED: Return None for the path if not found
    return (start, end, None)


class FinalTrainerNode(Node):
    def __init__(self):
        super().__init__('final_trainer_node')
        self.get_logger().info('Final Trainer Node started. Waiting for map...')
        
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_subscription = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
            
        self.grid = None
        self.training_complete = False

    def map_callback(self, msg):
        if self.training_complete:
            return

        self.get_logger().info('Map received! Processing grid...')
        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.get_logger().info(f'Grid processed. Shape: {self.grid.shape}. Starting training.')
        
        self.training_complete = True
        self.run_training()

    def run_training(self):
        self.get_logger().info("--- Generating Training Data using Parallel Processing ---")
        
        free_cells = list(zip(*np.where(self.grid <= 50))) 
        num_samples = 4000
        tasks = [(self.grid, start, end) for start, end in [random.sample(free_cells, 2) for _ in range(num_samples)]]

        num_cores = cpu_count()
        self.get_logger().info(f"Distributing {num_samples} A* searches across {num_cores} CPU cores...")
        
        start_time = time.time()
        X_data, y_data = [], []

        # <<< NEW: Use imap_unordered with tqdm for a live progress bar ---
        with Pool(processes=num_cores) as pool:
            # Wrap the parallel process with tqdm
            for start_cell, goal_cell, path in tqdm(pool.imap_unordered(run_a_star_worker, tasks), total=len(tasks), desc="Generating Data"):
                if path:
                    features = [start_cell[0], start_cell[1], goal_cell[0], goal_cell[1]]
                    cost = len(path) - 1.0
                    X_data.append(features)
                    y_data.append(cost)

        end_time = time.time()
        self.get_logger().info(f"Parallel data generation finished in {end_time - start_time:.2f} seconds.")
        self.get_logger().info(f"Generated {len(X_data)} valid samples.")
        
        if not X_data:
            self.get_logger().error("No data was generated. Cannot train model.")
            return

        # --- Model Training and Evaluation (Unchanged) ---
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
        self.get_logger().info("\n--- Training the ML Regression Model ---")
        ml_heuristic_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        ml_heuristic_model.fit(X_train, y_train)
        self.get_logger().info("Model training completed.")

        model_filename = 'heuristic_model.pkl'
        self.get_logger().info(f"\n--- Saving model to {model_filename} ---")
        with open(model_filename, 'wb') as file:
            pickle.dump(ml_heuristic_model, file)
        self.get_logger().info("Model saved successfully!")

        self.get_logger().info("\n--- Evaluating The Model on the Test Set ---")
        y_pred = ml_heuristic_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        self.get_logger().info(f"Mean Absolute Error (MAE): {mae:.4f}")
        self.get_logger().info(f"R-squared (R2): {r2:.4f}")

        plt.figure(figsize=(10, 7))
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', lw=2)
        plt.title('ML Heuristic Performance')
        plt.xlabel('Actual Path Cost (Ground Truth)')
        plt.ylabel('Predicted Path Cost (ML Model)')
        plt.grid(True)
        plt.savefig("model_performance.png")
        self.get_logger().info("Performance plot saved to model_performance.png")
        
        self.get_logger().info("\n\n *** TRAINING COMPLETE. You can now stop this node (Ctrl+C). *** \n")
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    trainer_node = FinalTrainerNode()
    try:
        rclpy.spin(trainer_node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if rclpy.ok() and hasattr(trainer_node, 'context') and trainer_node.context.is_valid():
            trainer_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()