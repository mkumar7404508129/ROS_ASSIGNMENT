# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Odometry
# import time
# import math
# import numpy as np

# class RectangularMotionNode(Node):
#     def __init__(self):
#         super().__init__('rectangular_motion_node')
#         self.get_logger().info('Rectangular motion node has started.')

#         # Publishers and Subscribers
#         self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.odometry_subscriber = self.create_subscription(
#             Odometry,
#             '/odom',
#             self.odometry_callback,
#             10)
        
#         # State machine parameters
#         self.current_state = 'moving_forward'
#         self.cycle_count = 0
        
#         # Motion parameters
#         self.linear_speed = 0.5  # m/s
#         self.angular_speed = 2.0 # rad/s (positive for anticlockwise turn)
#         self.target_distance = 8.0 # meters
#         self.target_angle = math.pi / 2.0 # 90 degrees anticlockwise
        
#         # Odom variables
#         self.current_pos = np.array([0.0, 0.0])
#         self.current_yaw = 0.0
#         self.start_pos = np.array([0.0, 0.0])
#         self.start_yaw = 0.0
#         self.odom_received = False
        
#         # Main control loop timer
#         self.timer = self.create_timer(0.1, self.timer_callback)

#     def odometry_callback(self, msg):
#         """Callback to update the robot's position and orientation from odometry."""
#         pos = msg.pose.pose.position
#         ori = msg.pose.pose.orientation
        
#         self.current_pos = np.array([pos.x, pos.y])
        
#         # Convert quaternion to Euler angles (yaw)
#         x = ori.x
#         y = ori.y
#         z = ori.z
#         w = ori.w
#         t3 = +2.0 * (w * z + x * y)
#         t4 = +1.0 - 2.0 * (y * y + z * z)
#         self.current_yaw = math.atan2(t3, t4)
        
#         if not self.odom_received:
#             self.start_pos = self.current_pos
#             self.start_yaw = self.current_yaw
#             self.odom_received = True

#     def timer_callback(self):
#         """Main control loop that executes the state machine."""
#         if not self.odom_received:
#             self.get_logger().warn('Waiting for odometry data...')
#             return

#         msg = Twist()
#         distance_moved = np.linalg.norm(self.current_pos - self.start_pos)
#         angle_turned = self.current_yaw - self.start_yaw

#         # Normalize angle to be within [-pi, pi]
#         if angle_turned > math.pi:
#             angle_turned -= 2 * math.pi
#         elif angle_turned < -math.pi:
#             angle_turned += 2 * math.pi
        
#         self.get_logger().info(f"State: {self.current_state}, Distance: {distance_moved:.2f}m, Angle: {angle_turned:.2f}rad")

#         if self.current_state == 'moving_forward':
#             if distance_moved >= self.target_distance:
#                 # Stop motion and transition to turning
#                 msg.linear.x = 0.0
#                 msg.angular.z = 0.0
#                 self.cmd_vel_publisher.publish(msg)
                
#                 self.current_state = 'turning'
#                 self.start_yaw = self.current_yaw # Reset reference yaw
#                 self.get_logger().info('Moved 8m. Stopping and preparing to turn.')
#             else:
#                 msg.linear.x = self.linear_speed
#                 msg.angular.z = 0.0
#                 self.cmd_vel_publisher.publish(msg)

#         elif self.current_state == 'turning':
#             if abs(angle_turned) >= abs(self.target_angle):
#                 # Stop turning and transition to moving forward
#                 msg.linear.x = 0.0
#                 msg.angular.z = 0.0
#                 self.cmd_vel_publisher.publish(msg)
                
#                 self.current_state = 'moving_forward'
#                 self.start_pos = self.current_pos # Reset reference position
#                 self.cycle_count += 1
#                 self.get_logger().info(f'Turned 90 deg. Stopping and preparing to move. Cycle: {self.cycle_count}')
#             else:
#                 msg.linear.x = 0.0
#                 msg.angular.z = self.angular_speed
#                 self.cmd_vel_publisher.publish(msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = RectangularMotionNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32 # Import the message type for the cycle count
import time
import math
import numpy as np

class RectangularMotionNode(Node):
    def __init__(self):
        super().__init__('rectangular_motion_node')
        self.get_logger().info('Rectangular motion node has started.')

        # Publishers and Subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # New publisher for the cycle count
        self.cycle_count_publisher = self.create_publisher(Int32, '/cycle_count', 10)
        self.odometry_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10)
        
        # State machine parameters
        self.current_state = 'moving_forward'
        self.cycle_count = 0
        self.target_cycles = 5 # Define the number of cycles to complete
        
        # Motion parameters
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 1.0 # rad/s (negative for clockwise turn)
        self.target_distance = 8.0 # meters
        self.target_angle = math.pi / 2.0 # 90 degrees clockwise
        
        # Odom variables
        self.current_pos = np.array([0.0, 0.0])
        self.current_yaw = 0.0
        self.start_pos = np.array([0.0, 0.0])
        self.start_yaw = 0.0
        self.odom_received = False
        
        # Main control loop timer
        self.timer = self.create_timer(0.1, self.timer_callback)

    def odometry_callback(self, msg):
        """Callback to update the robot's position and orientation from odometry."""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        
        self.current_pos = np.array([pos.x, pos.y])
        
        # Convert quaternion to Euler angles (yaw)
        x = ori.x
        y = ori.y
        z = ori.z
        w = ori.w
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        self.current_yaw = math.atan2(t3, t4)
        
        if not self.odom_received:
            self.start_pos = self.current_pos
            self.start_yaw = self.current_yaw
            self.odom_received = True

    def timer_callback(self):
        """Main control loop that executes the state machine."""
        if not self.odom_received:
            self.get_logger().warn('Waiting for odometry data...')
            return
        
        # Stop the node after the target number of cycles is reached
        if self.cycle_count >= self.target_cycles:
            self.get_logger().info(f'Completed {self.target_cycles} cycles. Shutting down node.')
            self.stop_robot()
            self.destroy_node()
            rclpy.shutdown()
            return

        msg = Twist()
        distance_moved = np.linalg.norm(self.current_pos - self.start_pos)
        angle_turned = self.current_yaw - self.start_yaw

        # Normalize angle to be within [-pi, pi]
        if angle_turned > math.pi:
            angle_turned -= 2 * math.pi
        elif angle_turned < -math.pi:
            angle_turned += 2 * math.pi
        
        self.get_logger().info(f"State: {self.current_state}, Cycle: {self.cycle_count}, Distance: {distance_moved:.2f}m, Angle: {angle_turned:.2f}rad")

        if self.current_state == 'moving_forward':
            if distance_moved >= self.target_distance:
                # Stop motion and transition to turning
                self.stop_robot()
                
                self.current_state = 'turning'
                self.start_yaw = self.current_yaw # Reset reference yaw
                self.get_logger().info('Moved 8m. Stopping and preparing to turn.')
            else:
                msg.linear.x = self.linear_speed
                msg.angular.z = 0.0
                self.cmd_vel_publisher.publish(msg)

        elif self.current_state == 'turning':
            if abs(angle_turned) >= abs(self.target_angle):
                # Stop turning and transition to moving forward
                self.stop_robot()
                
                self.current_state = 'moving_forward'
                self.start_pos = self.current_pos # Reset reference position
                self.cycle_count += 1

                # Publish the updated cycle count
                cycle_msg = Int32()
                cycle_msg.data = self.cycle_count
                self.cycle_count_publisher.publish(cycle_msg)

                self.get_logger().info(f'Turned 90 deg. Stopping and preparing to move. Cycle: {self.cycle_count}')
            else:
                msg.linear.x = 0.0
                msg.angular.z = self.angular_speed
                self.cmd_vel_publisher.publish(msg)

    def stop_robot(self):
        """Publishes a zero Twist message to stop the robot."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RectangularMotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
