#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosbag2_py import SequentialReader, StorageFilter
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time as TimeMsg

def quaternion_to_yaw(q):
    """Converts a geometry_msgs.msg.Quaternion to a yaw angle in radians."""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def main():
    rclpy.init()
    
    # --- Configuration ---
    bag_dir = 'rosbag2_2025_08_02-20_44_51'
    odom_topic = '/odom'
    amcl_pose_topic = '/amcl_pose'

    try:
        reader = SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader.open(storage_options, converter_options)
        
    except Exception as e:
        print(f"Error opening rosbag: {e}")
        print("Please ensure the bag directory exists and contains a .db3 file.")
        return

    # Filter to only read from our topics of interest
    topics_filter = StorageFilter(topics=[odom_topic, amcl_pose_topic])
    reader.set_filter(topics_filter)

    odom_data = []
    amcl_data = []

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    print("Reading data from rosbag...")

    while reader.has_next():
        (topic, data, t_ns) = reader.read_next()
        msg_type = type_map[topic]
        
        if topic == odom_topic:
            msg = deserialize_message(data, Odometry)
            odom_data.append({
                'timestamp': t_ns,
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'yaw': quaternion_to_yaw(msg.pose.pose.orientation)
            })
        elif topic == amcl_pose_topic:
            msg = deserialize_message(data, PoseStamped)
            amcl_data.append({
                'timestamp': t_ns,
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'yaw': quaternion_to_yaw(msg.pose.orientation)
            })

    print(f"Read {len(odom_data)} odom messages and {len(amcl_data)} amcl_pose messages.")
    
    if not odom_data or not amcl_data:
        print("Not enough data to plot. Please ensure both topics were recorded.")
        return

    odom_data = pd.DataFrame(odom_data)
    amcl_data = pd.DataFrame(amcl_data)

    odom_data['timestamp_s'] = odom_data['timestamp'] / 1e9
    amcl_data['timestamp_s'] = amcl_data['timestamp'] / 1e9
    
    amcl_data = amcl_data.set_index('timestamp_s')
    odom_data = odom_data.set_index('timestamp_s')

    aligned_amcl = amcl_data.reindex(odom_data.index, method='nearest')

    error_x = odom_data['x'] - aligned_amcl['x']
    error_y = odom_data['y'] - aligned_amcl['y']
    translational_error = np.sqrt(error_x**2 + error_y**2)

    error_yaw = odom_data['yaw'] - aligned_amcl['yaw']
    rotational_error = np.arctan2(np.sin(error_yaw), np.cos(error_yaw))

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Robot Paths
    # Using .values to get the underlying numpy array
    ax1.plot(odom_data['x'].values, odom_data['y'].values, 'b-', label='Ground Truth Path')
    ax1.plot(aligned_amcl['x'].values, aligned_amcl['y'].values, 'r--', label='MCL Estimated Path')
    ax1.set_title('Robot Path: Ground Truth vs. MCL Estimate')
    ax1.set_xlabel('X position (m)')
    ax1.set_ylabel('Y position (m)')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    # Plot 2: Errors Over Time
    # Using .values to get the underlying numpy array
    ax2.plot(odom_data.index.values, translational_error.values, 'g-', label='Translational Error (m)')
    ax2.plot(odom_data.index.values, np.degrees(np.abs(rotational_error.values)), 'm-', label='Rotational Error (deg)')
    ax2.set_title('Localization Accuracy Over Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()