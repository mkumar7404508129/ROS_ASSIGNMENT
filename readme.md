# ROS Navigation and Simulation Project

This repository contains a collection of ROS packages for robot navigation, SLAM (Simultaneous Localization and Mapping), and simulation. The project uses PyBullet for simulation and implements several popular navigation and localization algorithms.

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Prerequisites](#-prerequisites)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [A Note on .gitignore](#-a-note-on-gitignore)

## 🚀 Project Overview

This ROS workspace is designed to simulate a robot in a PyBullet environment and test various navigation and localization algorithms.

### Key Packages

- **pybullet_sim_pkg**: Contains the PyBullet simulation environment, including the robot model and nodes to control its motion.
- **astar_pkg / astar_ml_pkg**: Implements the A* pathfinding algorithm.
- **EKF_SLAM_pkg**: An implementation of Simultaneous Localization and Mapping using an Extended Kalman Filter.
- **MCL_pkg**: An implementation of Monte Carlo Localization (also known as a particle filter) for tracking the robot's pose.

## ✅ Prerequisites

Before you begin, ensure you have the following installed on your system:

- **ROS**: (e.g., ROS Noetic or ROS 2 Foxy). These instructions assume a Catkin workspace (ROS 1).
- **Python 3** and pip.
- **PyBullet**: The physics simulator.

```bash
pip install pybullet
```

- **Git**: For cloning the repository.

## 🛠️ Setup & Installation

To set up this project on a new machine after cloning, follow these steps. This process will create a new ROS workspace, clone the project into it, install dependencies, and build the packages.

### Create a new Catkin Workspace:

```bash
mkdir -p ~/ros_ws/src
cd ~/ros_ws/
catkin_make
```

### Clone the Repository:

Navigate to the `src` directory of your new workspace and clone this repository.

```bash
cd ~/ros_ws/src
git clone <your-repository-url-here> .
```

> Replace `<your-repository-url-here>` with your actual Git repository URL. The `.` at the end clones it directly into the `src` folder.

### Install Dependencies:

Use `rosdep` to install any package dependencies listed in the `package.xml` files.

```bash
cd ~/ros_ws
rosdep install --from-paths src --ignore-src -r -y
```

### Build the Workspace:

Use `catkin_make` (for ROS 1) or `colcon build` (for ROS 2) to compile all the packages.

```bash
cd ~/ros_ws
catkin_make
```

### Source the Workspace:

Finally, source the `setup.bash` file to add this workspace's packages to your ROS environment.

```bash
source ~/ros_ws/devel/setup.bash
```

> Tip: Add this command to your `~/.bashrc` file to automatically source it every time you open a new terminal.

## ▶️ Usage

To run the simulation and navigation, you will typically launch one or more nodes.

### Launch the Simulator:

```bash
rosrun pybullet_sim_pkg robot_simulator_node.py
```

### Run a Navigation/Localization Node:

In a new terminal, run one of the algorithm nodes (remember to source your workspace first!).

```bash
# Example for EKF SLAM
rosrun EKF_SLAM_pkg ekf_slam_node

# Example for MCL
rosrun MCL_pkg mcl_node
```

> Note: The exact node names and launch files might differ. Update as needed.

## 📝 A Note on .gitignore

The `.gitignore` file is crucial for keeping your repository clean and efficient. It tells Git which files and directories it should not track.

### Here’s a breakdown of the rules you've included:

- `**/build/`, `**/install/`, `**/log/`: These directories are generated automatically when you build your ROS workspace (`catkin_make`). They contain compiled code, executables, and log files. They don't need to be version-controlled because they can be regenerated from the source code at any time. Including them would bloat the repository with unnecessary files.
- `.DS_Store`: This is a metadata file created automatically by macOS. It's specific to a user's machine and has no value to other collaborators.
- `*.bag`, `*.db3`: These are ROS bag files, which are used to record and play back ROS message data. They can become very large and are typically not stored in a Git repository.
- `*.png`: You've added this to ignore PNG files. While you might want to track some images (like the `rosgraph.png`), this rule prevents accidental check-ins of other images, like maps generated during a SLAM run.

If you want to keep `rosgraph.png`, you can add an exception to your `.gitignore` file like this:

```gitignore
!rosgraph.png
```

By ignoring these files, you ensure that anyone who clones your repository gets only the essential source code needed to build and run the project.
