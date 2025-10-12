
from __future__ import division, print_function
import rospy
import numpy as np
from collections import deque
from geometry_msgs.msg import PointStamped, Point
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import Header
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import actionlib
import tf2_ros
import tf2_geometry_msgs
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class SimpleTrajectoryPredictor:
    def __init__(self):
        rospy.init_node('simple_trajectory_predictor', anonymous=True)
        
        # Parameters
        self.window_size = 10  # Number of points to keep for prediction
        self.prediction_steps = 5  # Number of future steps to predict
        self.prediction_time_step = 0.1  # Time step for predictions (seconds)
        self.goal_threshold = 0.5  # Minimum distance to send new goal (meters)
        
        # Data storage
        self.trajectory_points = deque(maxlen=self.window_size)
        self.timestamps = deque(maxlen=self.window_size)
        self.last_goal_position = None
        
        # TF2 Buffer and Listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers and Publishers
        self.image_detections_sub = rospy.Subscriber('/raw_bodies', PoseArray, self.image_detections_callback)
        self.prediction_pub = rospy.Publisher('/predicted_trajectory', PoseArray, queue_size=10)
        
        # Move base action client for robot navigation
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base action server")
        
        rospy.loginfo("Simple Trajectory Predictor initialized")
    
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
    
    def image_detections_callback(self, msg):
        """
        Callback function for receiving person coordinates from PoseArray and transforming to odom
        """
        try:
            # Check if we have any poses in the array
            if not msg.poses:
                rospy.logwarn("Received empty PoseArray")
                return
            
            # Use the first pose in the array (you can modify this logic if needed)
            pose_base_footprint = msg.poses[0]
            
            # Transform pose from base_footprint to odom frame
            pose_odom = self.transform_pose_to_odom(pose_base_footprint, "base_footprint")
            
            if pose_odom is None:
                rospy.logwarn("Failed to transform pose to odom frame")
                return
            
            # Extract coordinates and timestamp (now in odom frame)
            x = pose_odom.position.x
            y = pose_odom.position.y
            timestamp = msg.header.stamp.to_sec()
            
            # Store the point and timestamp
            self.trajectory_points.append((x, y))
            self.timestamps.append(timestamp)
            
            rospy.loginfo("Received point in odom: ({:.2f}, {:.2f}) at time {:.3f}".format(x, y, timestamp))
            
            # Predict trajectory if we have enough points
            if len(self.trajectory_points) >= 3:
                predicted_trajectory = self.predict_trajectory()
                #self.publish_prediction(predicted_trajectory)
                
                # Send robot to predicted position
                if predicted_trajectory:
                    self.send_robot_to_prediction(predicted_trajectory)
                
        except Exception as e:
            rospy.logerr("Error in callback: {}".format(str(e)))
    
    def predict_trajectory(self):
        """
        Predict future trajectory points using linear regression
        """
        if len(self.trajectory_points) < 3:
            return []
        
        # Convert to numpy arrays
        points = np.array(list(self.trajectory_points))
        times = np.array(list(self.timestamps))
        
        # Normalize times to start from 0
        times = times - times[0]
        
        # Fit linear models for x and y coordinates
        try:
            # Linear fit: coordinate = a * time + b
            x_coeffs = np.polyfit(times, points[:, 0], 1)
            y_coeffs = np.polyfit(times, points[:, 1], 1)
            
            # Generate future time points
            last_time = times[-1]
            future_times = []
            for i in range(1, self.prediction_steps + 1):
                future_times.append(last_time + i * self.prediction_time_step)
            
            # Predict future points
            predicted_points = []
            for t in future_times:
                x_pred = x_coeffs[0] * t + x_coeffs[1]
                y_pred = y_coeffs[0] * t + y_coeffs[1]
                predicted_points.append((x_pred, y_pred))
            
            return predicted_points
            
        except Exception as e:
            rospy.logerr("Error in trajectory prediction: {}".format(str(e)))
            return []
    
    def polynomial_predict_trajectory(self):
        """
        Alternative method: Predict using polynomial fitting (more sophisticated)
        """
        if len(self.trajectory_points) < 4:
            return []
        
        points = np.array(list(self.trajectory_points))
        times = np.array(list(self.timestamps))
        times = times - times[0]
        
        try:
            # Polynomial fit (degree 2)
            x_coeffs = np.polyfit(times, points[:, 0], min(2, len(times)-1))
            y_coeffs = np.polyfit(times, points[:, 1], min(2, len(times)-1))
            
            last_time = times[-1]
            predicted_points = []
            
            for i in range(1, self.prediction_steps + 1):
                t = last_time + i * self.prediction_time_step
                x_pred = np.polyval(x_coeffs, t)
                y_pred = np.polyval(y_coeffs, t)
                predicted_points.append((x_pred, y_pred))
            
            return predicted_points
            
        except Exception as e:
            rospy.logerr("Error in polynomial prediction: {}".format(str(e)))
            return []
    
    def publish_prediction(self, predicted_points):
        """
        Publish predicted trajectory as PoseArray
        """
        if not predicted_points:
            return
        
        try:
            pose_array = PoseArray()
            pose_array.header = Header()
            pose_array.header.stamp = rospy.Time.now()
            pose_array.header.frame_id = "odom"
            
            for x, y in predicted_points:
                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = 0.0
                pose.orientation.w = 1.0  # Default orientation
                pose_array.poses.append(pose)
            
            self.prediction_pub.publish(pose_array)
            rospy.loginfo("Published {} predicted points".format(len(predicted_points)))
            
        except Exception as e:
            rospy.logerr("Error publishing prediction: {}".format(str(e)))
    
    def send_robot_to_prediction(self, predicted_points):
        """
        Send the robot to the predicted future position
        """
        if not predicted_points:
            return
            
        try:
            # Use the first predicted point as the goal
            target_x, target_y = predicted_points[0]
            
            # Check if we should send a new goal (avoid sending too many goals)
            if self.last_goal_position is not None:
                dist = np.sqrt((target_x - self.last_goal_position[0])**2 + 
                              (target_y - self.last_goal_position[1])**2)
                if dist < self.goal_threshold:
                    return  # Don't send new goal if too close to last one
            
            # Create and send move_base goal
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "odom"  # Now using odom frame
            goal.target_pose.header.stamp = rospy.Time.now()
            
            # Set position
            goal.target_pose.pose.position.x = target_x
            goal.target_pose.pose.position.y = target_y
            goal.target_pose.pose.position.z = 0.0
            
            # Set orientation (face towards the predicted trajectory direction)
            if len(predicted_points) > 1:
                # Calculate direction from first to second predicted point
                dx = predicted_points[1][0] - predicted_points[0][0]
                dy = predicted_points[1][1] - predicted_points[0][1]
                yaw = np.arctan2(dy, dx)
                
                # Convert yaw to quaternion
                goal.target_pose.pose.orientation.x = 0.0
                goal.target_pose.pose.orientation.y = 0.0
                goal.target_pose.pose.orientation.z = np.sin(yaw / 2.0)
                goal.target_pose.pose.orientation.w = np.cos(yaw / 2.0)
            else:
                # Default orientation
                goal.target_pose.pose.orientation.w = 1.0
            
            # Send the goal
            self.move_base_client.send_goal(goal)
            self.last_goal_position = (target_x, target_y)
            
            rospy.loginfo("Sent robot to predicted position: ({:.2f}, {:.2f})".format(target_x, target_y))
            
        except Exception as e:
            rospy.logerr("Error sending robot goal: {}".format(str(e)))
    
    def cancel_robot_goal(self):
        """
        Cancel the current robot navigation goal
        """
        try:
            self.move_base_client.cancel_goal()
            rospy.loginfo("Cancelled robot navigation goal")
        except Exception as e:
            rospy.logerr("Error cancelling robot goal: {}".format(str(e)))
    
    def get_robot_goal_status(self):
        """
        Get the current status of the robot navigation goal
        """
        try:
            state = self.move_base_client.get_state()
            if state == GoalStatus.PENDING:
                return "PENDING"
            elif state == GoalStatus.ACTIVE:
                return "ACTIVE"
            elif state == GoalStatus.SUCCEEDED:
                return "SUCCEEDED"
            elif state == GoalStatus.ABORTED:
                return "ABORTED"
            elif state == GoalStatus.REJECTED:
                return "REJECTED"
            else:
                return "UNKNOWN"
        except Exception as e:
            rospy.logerr("Error getting robot goal status: {}".format(str(e)))
            return "ERROR"

def main():
    """
    Main function to run the trajectory predictor
    """
    try:
        predictor = SimpleTrajectoryPredictor()
        rospy.loginfo("Trajectory predictor is running...")
        rospy.loginfo("Waiting for person coordinates on /raw_bodies topic...")
        
        # Keep the node running
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Trajectory predictor node interrupted")
    except Exception as e:
        rospy.logerr("Error in main: {}".format(str(e)))

if __name__ == "__main__":
    main()