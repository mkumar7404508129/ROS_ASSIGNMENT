#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np

class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher_node')
        self.get_logger().info("Map Publisher Node started. Publishing a static hardcoded map.")
        self.map_pub = self.create_publisher(OccupancyGrid, 'map', 1)

        # Hardcoded map data (same as in mcl_node)
        self.occupancy_grid = self.create_occupancy_grid()

        # Publish the map immediately and then again after a short delay
        self.map_timer = self.create_timer(1.0, self.publish_map_once)
        self.published = False

    def create_occupancy_grid(self):
        """Creates a simple hardcoded occupancy grid for this example."""
        map_size = 100
        grid = np.ones((map_size, map_size), dtype=np.int8) * -1  # -1 for unknown

        map_resolution = 0.1
        map_origin_x = -5.0
        map_origin_y = -5.0

        grid[:, 0] = 100
        grid[:, map_size - 1] = 100
        grid[0, :] = 100
        grid[map_size - 1, :] = 100

        obs_start = int(map_size / 2) - 10
        obs_end = int(map_size / 2) + 10
        grid[obs_start:obs_end, obs_start:obs_end] = 100

        return {
            'data': grid,
            'info': {
                'resolution': map_resolution,
                'width': map_size,
                'height': map_size,
                'origin': {'position': {'x': map_origin_x, 'y': map_origin_y, 'z': 0.0}}
            }
        }

    def publish_map_once(self):
        if not self.published:
            self.get_logger().info("Publishing OccupancyGrid message.")
            msg = OccupancyGrid()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.info.map_load_time = msg.header.stamp
            msg.info.resolution = self.occupancy_grid['info']['resolution']
            msg.info.width = self.occupancy_grid['info']['width']
            msg.info.height = self.occupancy_grid['info']['height']
            msg.info.origin.position.x = self.occupancy_grid['info']['origin']['position']['x']
            msg.info.origin.position.y = self.occupancy_grid['info']['origin']['position']['y']
            msg.data = self.occupancy_grid['data'].flatten().tolist()
            self.map_pub.publish(msg)
            self.published = True
            self.map_timer.cancel() # Stop the timer after publishing

def main(args=None):
    rclpy.init(args=args)
    map_publisher = MapPublisher()
    rclpy.spin(map_publisher)
    map_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()