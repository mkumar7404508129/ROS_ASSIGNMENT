# ROS 2 PyBullet Robot Simulation (`pybullet_sim_pkg`)

This ROS 2 package provides a simple robot simulation environment using the PyBullet physics engine. It is designed to simulate a differential drive robot in a custom world and publish standard ROS 2 messages for sensor data and odometry.

The project includes two main nodes:
1.  **`robot_simulator_node`**: The core simulation environment.
2.  **`rectangular_motion_node`**: An example controller to move the robot in a pattern.

## System Architecture

Here is a simplified diagram of how the nodes and key topics interact:

![Diagram description](../../simulation_pkgs/simulation_node.png)
