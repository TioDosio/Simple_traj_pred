#!/usr/bin/env python
import rospy
import actionlib
import math
import threading
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry
import tf2_ros
import tf2_geometry_msgs

def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    Compatible replacement for tf.transformations.quaternion_from_euler
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [x, y, z, w]

class AdaptiveRobotNavigator:
    def __init__(self, topic_name="/raw_bodies", distance_threshold=1.0, max_distance_threshold=5.0):
        """
        Initialize the adaptive robot navigator.
        
        Args:
            topic_name (str): ROS topic name to subscribe to for pose messages
            distance_threshold (float): Distance threshold in meters to consider "close enough"
            max_distance_threshold (float): Maximum distance threshold to filter out hallucinated poses
        """
        self.distance_threshold = distance_threshold
        self.max_distance_threshold = max_distance_threshold
        self.current_target = None
        self.previous_distance = None
        self.is_moving = False
        self.lock = threading.Lock()
        self.pose_count = 0  # Counter for pose messages
        self.robot_position = None  # Current robot position from odometry
        self.robot_orientation = None  # Current robot orientation from odometry
        
        # Initialize ROS node
        if not rospy.get_node_uri():
            rospy.init_node('adaptive_robot_navigator', anonymous=True)
        
        # TF2 Buffer and Listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Setup action client for move_base
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        rospy.loginfo("Waiting for move_base action server...")
        if not self.client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("move_base action server not available")
            # Continue anyway to listen for poses
        else:
            rospy.loginfo("Connected to move_base action server successfully")
            # Check if there's any current goal
            state = self.client.get_state()
            rospy.loginfo("Current move_base state: {}".format(state))
        
        # Subscribe to pose array topic and odometry
        self.pose_subscriber = rospy.Subscriber(topic_name, PoseArray, self.pose_callback)
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.loginfo("Subscribed to {} with distance threshold: {}m".format(topic_name, distance_threshold))
        rospy.loginfo("Subscribed to /odom for robot position tracking")
        rospy.loginfo("Waiting for pose messages... Use 'rostopic echo /raw_bodies' to check if messages are being published")
    
    def odom_callback(self, msg):
        """
        Callback function for odometry messages to track robot position.
        """
        with self.lock:
            self.robot_position = msg.pose.pose.position
            self.robot_orientation = msg.pose.pose.orientation
    
    def transform_pose_to_odom(self, pose, source_frame="base_footprint"):
        """
        Transform a pose from source_frame to odom frame
        """
        try:
            # Create a PoseStamped message
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = source_frame
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.pose = pose
            
            # Transform to odom frame
            transformed_pose = self.tf_buffer.transform(pose_stamped, "odom", rospy.Duration(1.0))
            return transformed_pose.pose
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Could not transform pose to odom: {}".format(str(e)))
            return None
    
    def pose_callback(self, msg):
        """
        Callback function for pose array messages.
        Assumes we want to track the first pose in the array.
        """
        if not msg.poses:
            rospy.logwarn("Received empty pose array!")
            return
        
        # Check if we have robot position from odometry
        if self.robot_position is None:
            rospy.logwarn("No robot position available from odometry yet")
            return
            
        # Get the first pose from the array and transform to odom frame
        target_pose_base_footprint = msg.poses[0]
        
        # Transform pose from base_footprint to odom frame
        target_pose_odom = self.transform_pose_to_odom(target_pose_base_footprint, "base_footprint")
        
        if target_pose_odom is None:
            rospy.logwarn("Failed to transform pose to odom frame")
            return
        
        # Extract target coordinates in odom frame
        target_x = target_pose_odom.position.x
        target_y = target_pose_odom.position.y
        
        # Calculate distance between robot and target (both in odom frame)
        with self.lock:
            if self.robot_position is not None:
                current_distance = math.sqrt(
                    (target_x - self.robot_position.x)**2 + 
                    (target_y - self.robot_position.y)**2
                )
            else:
                rospy.logwarn("Robot position not available")
                return
        
        # Only log every 10th pose message to reduce clutter
        self.pose_count += 1
        if self.pose_count % 10 == 1:
            rospy.loginfo("Target in odom: x={:.2f}, y={:.2f}, Robot: x={:.2f}, y={:.2f}, distance={:.2f}m".format(
                target_x, target_y, self.robot_position.x, self.robot_position.y, current_distance))
        
        # Check if the pose is too far away (likely hallucination)
        if current_distance > self.max_distance_threshold:
            rospy.logwarn("Target too far ({:.2f}m > {:.2f}m), ignoring".format(
                current_distance, self.max_distance_threshold))
            return
        
        with self.lock:
            # Check if we should start moving, stop, or continue
            if current_distance <= self.distance_threshold:
                # Target is close enough - stop if we're moving
                if self.is_moving:
                    rospy.loginfo("Target close enough ({:.2f}m <= {:.2f}m), stopping robot".format(
                        current_distance, self.distance_threshold))
                    if hasattr(self, 'client'):
                        self.client.cancel_goal()
                    self.is_moving = False
            else:
                # Target is far - start moving if we're not already
                if not self.is_moving:
                    rospy.loginfo("Target far enough ({:.2f}m > {:.2f}m), moving to target".format(
                        current_distance, self.distance_threshold))
                    # Use robot's current orientation instead of target's orientation
                    self.move_to_target(target_x, target_y, self.robot_orientation)
            
            self.previous_distance = current_distance
    
    def move_to_target(self, x, y, orientation, frame="odom"):
        """
        Send a goal to move_base to navigate to the target position.
        """
        try:
            rospy.loginfo("Sending navigation goal to: x={:.2f}, y={:.2f} in {} frame".format(x, y, frame))
            
            # Build goal
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = frame
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = x
            goal.target_pose.pose.position.y = y
            goal.target_pose.pose.position.z = 0.0  # Explicitly set z to 0
            goal.target_pose.pose.orientation = orientation  # Use robot's current orientation
            
            if hasattr(self, 'client') and self.client:
                self.client.send_goal(goal, done_cb=self.goal_done_callback, 
                                    feedback_cb=self.goal_feedback_callback)
                self.is_moving = True
                rospy.loginfo("Goal sent successfully")
                
                # Check goal status after a short delay
                rospy.Timer(rospy.Duration(1.0), self.check_goal_status, oneshot=True)
            else:
                rospy.logwarn("Move_base client not available, simulating movement")
                # Still set is_moving to True for testing the logic
                self.is_moving = True
            
        except Exception as e:
            rospy.logerr("Error sending goal: {}".format(e))
            self.is_moving = False
    
    def goal_done_callback(self, status, result):
        """
        Callback when goal is completed (success, failure, or cancelled).
        """
        rospy.loginfo("Goal completed with status: {}".format(status))
        if status == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Navigation goal reached successfully!")
        elif status == actionlib.GoalStatus.ABORTED:
            rospy.logwarn("Navigation goal was aborted")
        elif status == actionlib.GoalStatus.PREEMPTED:
            rospy.loginfo("Navigation goal was cancelled")
        else:
            rospy.logwarn("Navigation goal finished with status: {}".format(status))
        
        with self.lock:
            self.is_moving = False
    
    def goal_feedback_callback(self, feedback):
        """
        Callback for goal feedback (current robot position during navigation).
        """
        current_pos = feedback.base_position.pose.position
        rospy.loginfo("Robot moving - current position: x={:.2f}, y={:.2f}".format(
            current_pos.x, current_pos.y))
    
    def check_goal_status(self, event):
        """
        Check the current status of the goal after sending it.
        """
        if hasattr(self, 'client') and self.client:
            state = self.client.get_state()
            rospy.loginfo("Goal status after 1 second: {}".format(state))
            if state == actionlib.GoalStatus.PENDING:
                rospy.loginfo("Goal is pending...")
            elif state == actionlib.GoalStatus.ACTIVE:
                rospy.loginfo("Goal is active - robot should be moving")
            elif state == actionlib.GoalStatus.ABORTED:
                rospy.logwarn("Goal was aborted immediately - check for obstacles or invalid goal")
            elif state == actionlib.GoalStatus.REJECTED:
                rospy.logerr("Goal was rejected - check navigation stack configuration")
    
    def set_distance_threshold(self, new_threshold):
        """
        Update the distance threshold for stopping.
        
        Args:
            new_threshold (float): New distance threshold in meters
        """
        with self.lock:
            self.distance_threshold = new_threshold
    
    def set_max_distance_threshold(self, new_max_threshold):
        """
        Update the maximum distance threshold for filtering hallucinations.
        
        Args:
            new_max_threshold (float): New maximum distance threshold in meters
        """
        with self.lock:
            self.max_distance_threshold = new_max_threshold
    
    def stop_robot(self):
        """
        Stop the robot by canceling current goal.
        """
        with self.lock:
            if self.is_moving:
                if hasattr(self, 'client'):
                    self.client.cancel_goal()
                self.is_moving = False
    
    def get_status(self):
        """
        Get current status of the navigator.
        
        Returns:
            dict: Current status including distance threshold, moving state, etc.
        """
        with self.lock:
            return {
                'distance_threshold': self.distance_threshold,
                'max_distance_threshold': self.max_distance_threshold,
                'is_moving': self.is_moving,
                'previous_distance': self.previous_distance,
                'has_target': self.current_target is not None
            }
    
    def spin(self):
        """
        Keep the node running and processing callbacks.
        """
        rospy.spin()


def move_robot_to_coordinate(x, y, yaw=0.0, frame="base_footprint", timeout=60.0, cancel_after=None):
    try:
        # Initialize ROS node
        if not rospy.get_node_uri():
            rospy.init_node('robot_navigation', anonymous=True)
        
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        
        if not client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("move_base action server not available")
            return False
        
        # Build goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        q = quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        rospy.loginfo("Sending goal: x={:.2f}, y={:.2f}, yaw={:.2f}".format(x, y, yaw))
        client.send_goal(goal)

        # Optional: cancel goal after some time
        if cancel_after is not None:
            rospy.sleep(cancel_after)
            rospy.logwarn("Cancelling goal after {:.1f} seconds!".format(cancel_after))
            client.cancel_goal()

        # Wait for result
        finished = client.wait_for_result(rospy.Duration(timeout))
        if not finished:
            rospy.logwarn("Goal timed out after {} seconds.".format(timeout))
            client.cancel_goal()
            return False

        state = client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Successfully reached target!")
            return True
        else:
            rospy.logwarn("Failed or cancelled. State: {}".format(state))
            return False

    except Exception as e:
        rospy.logerr("Error in move_robot_to_coordinate: {}".format(e))
        return False


if __name__ == '__main__':
    try:
        # Configuration
        TOPIC_NAME = "/raw_bodies"  # Your actual topic name
        DISTANCE_THRESHOLD = 1.0    # Distance in meters to consider "close enough" (changed to 1m)
        MAX_DISTANCE_THRESHOLD = 5  # Maximum distance to accept (filter hallucinations)
        
        # Create and start the adaptive navigator
        navigator = AdaptiveRobotNavigator(
            topic_name=TOPIC_NAME,
            distance_threshold=DISTANCE_THRESHOLD,
            max_distance_threshold=MAX_DISTANCE_THRESHOLD
        )
        
        # Optional: You can change the thresholds dynamically
        # navigator.set_distance_threshold(0.3)     # Change to 30cm
        # navigator.set_max_distance_threshold(3.0) # Change max to 3m
        
        # Start listening for messages and processing navigation
        navigator.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation interrupted by user")
    except KeyboardInterrupt:
        rospy.loginfo("Navigation stopped by user")
    except Exception as e:
        rospy.logerr("Error in main: {}".format(e))
    
    # Legacy example using the original function
    # success = move_robot_to_coordinate(1.0, 0.0, yaw=0.0, cancel_after=5.0)
    # if success:
    #     print("Robot reached the target!")
    # else:
    #     print("Goal was cancelled or failed.")



