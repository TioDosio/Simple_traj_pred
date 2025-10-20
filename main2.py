
from __future__ import division, print_function
import rospy
import numpy as np
from collections import deque
from geometry_msgs.msg import PointStamped, Point
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_msgs.msg import TFMessage
import tf.transformations 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class SimpleTrajectoryPredictor:
    def __init__(self):
        rospy.init_node('simple_trajectory_predictor', anonymous=True)
        
        # Parameters
        self.window_size = 10  # Number of points to keep for prediction
        self.prediction_steps = 6  # Number of future steps to predict
        self.prediction_time_step = 0.2  # Time step for predictions (seconds)
        self.goal_threshold = 0.2  # Minimum distance to send new goal (meters)
        
        # Data storage
        self.trajectory_points = deque(maxlen=self.window_size)
        self.timestamps = deque(maxlen=self.window_size)
        
        # TF transformation storage
        self.base_footprint_to_odom_transform = None
        self.transform_timestamp = 0.0
        
        # Prediction tracking for table generation
        self.prediction_counter = 0
        self.ground_truth_data = {}  # Store ground truth for comparison
        
        # Subscribers and Publishers
        self.image_detections_sub = rospy.Subscriber('/raw_bodies', PoseArray, self.image_detections_callback)
        self.prediction_pub = rospy.Publisher('/predicted_trajectory', PoseArray, queue_size=10)
        self.visualization_pub = rospy.Publisher('/trajectory_visualization', MarkerArray, queue_size=10)
        self.tf_sub = rospy.Subscriber('/tf', TFMessage, self.tf_callback)
        
        rospy.loginfo("Simple Trajectory Predictor initialized")
    
    def tf_callback(self, msg):
        """
        Callback function for receiving transform messages from /tf topic
        """
        try:
            for transform in msg.transforms:
                # Look for base_footprint to odom transformation
                if (transform.child_frame_id == "base_footprint" and 
                    transform.header.frame_id == "odom"):
                    self.base_footprint_to_odom_transform = transform
                    self.transform_timestamp = transform.header.stamp.to_sec()
                    break
        except Exception as e:
            rospy.logerr("Error in tf callback: {}".format(str(e)))
    
    def transform_pose_to_odom(self, pose, source_frame="base_footprint"):
        """
        Transform a pose from source_frame to odom frame using stored tf transform
        """
        if self.base_footprint_to_odom_transform is None:
            rospy.logwarn("No transform available from {} to odom".format(source_frame))
            return None
            
        try:
            # Get the transformation
            transform = self.base_footprint_to_odom_transform.transform
            
            # Extract translation and rotation from transform
            trans_x = transform.translation.x
            trans_y = transform.translation.y
            trans_z = transform.translation.z
            
            rot_x = transform.rotation.x
            rot_y = transform.rotation.y
            rot_z = transform.rotation.z
            rot_w = transform.rotation.w
            
            # Convert quaternion to rotation matrix
            rotation_matrix = tf.transformations.quaternion_matrix([rot_x, rot_y, rot_z, rot_w])
            
            # Original position in base_footprint frame
            pos_x = pose.position.x
            pos_y = pose.position.y
            pos_z = pose.position.z
            
            # Apply rotation
            point_homogeneous = np.array([pos_x, pos_y, pos_z, 1.0])
            transformed_point = np.dot(rotation_matrix, point_homogeneous)
            
            # Apply translation
            final_x = transformed_point[0] + trans_x
            final_y = transformed_point[1] + trans_y
            final_z = transformed_point[2] + trans_z
            
            # Create transformed pose
            transformed_pose = Pose()
            transformed_pose.position.x = final_x
            transformed_pose.position.y = final_y
            transformed_pose.position.z = final_z
            
            # Keep original orientation (or transform it if needed)
            transformed_pose.orientation = pose.orientation
            
            return transformed_pose
            
        except Exception as e:
            rospy.logerr("Error transforming pose: {}".format(str(e)))
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
                
                # Uncomment the line below to enable RViz visualization
                self.publish_trajectory_visualization(predicted_trajectory)
                
                # Create and display prediction table
                current_pos = (x, y)
                self.create_prediction_table(predicted_trajectory, current_pos)
                
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
    
    def publish_trajectory_visualization(self, predicted_points):
        """
        Publish trajectory visualization to RViz with dots and lines for observed and predicted data
        Call this function to visualize trajectories in RViz
        """
        if not predicted_points and len(self.trajectory_points) == 0:
            return
            
        try:
            marker_array = MarkerArray()
            marker_id = 0
            current_time = rospy.Time.now()
            
            # === OBSERVED TRAJECTORY VISUALIZATION ===
            
            # 1. Observed points as dots (spheres)
            if len(self.trajectory_points) > 0:
                for i, (x, y) in enumerate(self.trajectory_points):
                    marker = Marker()
                    marker.header.frame_id = "odom"
                    marker.header.stamp = current_time
                    marker.ns = "observed_points"
                    marker.id = marker_id
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    
                    # Position
                    marker.pose.position.x = x
                    marker.pose.position.y = y
                    marker.pose.position.z = 0.1
                    marker.pose.orientation.w = 1.0
                    
                    # Scale (size of the dot)
                    marker.scale.x = 0.15
                    marker.scale.y = 0.15
                    marker.scale.z = 0.15
                    
                    # Color - Blue for observed points
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                    marker.color.a = 0.8
                    
                    marker.lifetime = rospy.Duration(1.0)
                    marker_array.markers.append(marker)
                    marker_id += 1
                
                # 2. Observed trajectory as a line
                if len(self.trajectory_points) > 1:
                    line_marker = Marker()
                    line_marker.header.frame_id = "odom"
                    line_marker.header.stamp = current_time
                    line_marker.ns = "observed_line"
                    line_marker.id = marker_id
                    line_marker.type = Marker.LINE_STRIP
                    line_marker.action = Marker.ADD
                    
                    # Line properties
                    line_marker.scale.x = 0.05  # Line width
                    
                    # Color - Blue for observed line
                    line_marker.color.r = 0.0
                    line_marker.color.g = 0.0
                    line_marker.color.b = 1.0
                    line_marker.color.a = 0.6
                    
                    # Add all observed points to the line
                    for x, y in self.trajectory_points:
                        point = Point()
                        point.x = x
                        point.y = y
                        point.z = 0.05
                        line_marker.points.append(point)
                    
                    line_marker.lifetime = rospy.Duration(1.0)
                    marker_array.markers.append(line_marker)
                    marker_id += 1
            
            # === PREDICTED TRAJECTORY VISUALIZATION ===
            
            # 3. Predicted points as dots (spheres)
            if predicted_points:
                for i, (x, y) in enumerate(predicted_points):
                    marker = Marker()
                    marker.header.frame_id = "odom"
                    marker.header.stamp = current_time
                    marker.ns = "predicted_points"
                    marker.id = marker_id
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    
                    # Position
                    marker.pose.position.x = x
                    marker.pose.position.y = y
                    marker.pose.position.z = 0.1
                    marker.pose.orientation.w = 1.0
                    
                    # Scale (slightly larger for predicted points)
                    marker.scale.x = 0.20
                    marker.scale.y = 0.20
                    marker.scale.z = 0.20
                    
                    # Color - Red for predicted points
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.8
                    
                    marker.lifetime = rospy.Duration(1.0)
                    marker_array.markers.append(marker)
                    marker_id += 1
                
                # 4. Predicted trajectory as a line
                if len(predicted_points) > 1:
                    pred_line_marker = Marker()
                    pred_line_marker.header.frame_id = "odom"
                    pred_line_marker.header.stamp = current_time
                    pred_line_marker.ns = "predicted_line"
                    pred_line_marker.id = marker_id
                    pred_line_marker.type = Marker.LINE_STRIP
                    pred_line_marker.action = Marker.ADD
                    
                    # Line properties
                    pred_line_marker.scale.x = 0.08  # Slightly thicker line for predictions
                    
                    # Color - Red for predicted line
                    pred_line_marker.color.r = 1.0
                    pred_line_marker.color.g = 0.0
                    pred_line_marker.color.b = 0.0
                    pred_line_marker.color.a = 0.6
                    
                    # Add all predicted points to the line
                    for x, y in predicted_points:
                        point = Point()
                        point.x = x
                        point.y = y
                        point.z = 0.05
                        pred_line_marker.points.append(point)
                    
                    pred_line_marker.lifetime = rospy.Duration(1.0)
                    marker_array.markers.append(pred_line_marker)
                    marker_id += 1
                
                # 5. Connection line from last observed to first predicted point
                if len(self.trajectory_points) > 0:
                    last_observed = list(self.trajectory_points)[-1]
                    first_predicted = predicted_points[0]
                    
                    connection_marker = Marker()
                    connection_marker.header.frame_id = "odom"
                    connection_marker.header.stamp = current_time
                    connection_marker.ns = "connection_line"
                    connection_marker.id = marker_id
                    connection_marker.type = Marker.LINE_STRIP
                    connection_marker.action = Marker.ADD
                    
                    # Line properties
                    connection_marker.scale.x = 0.03
                    
                    # Color - Green for connection
                    connection_marker.color.r = 0.0
                    connection_marker.color.g = 1.0
                    connection_marker.color.b = 0.0
                    connection_marker.color.a = 0.8
                    
                    # Add connection points
                    point1 = Point()
                    point1.x = last_observed[0]
                    point1.y = last_observed[1]
                    point1.z = 0.05
                    
                    point2 = Point()
                    point2.x = first_predicted[0]
                    point2.y = first_predicted[1]
                    point2.z = 0.05
                    
                    connection_marker.points.append(point1)
                    connection_marker.points.append(point2)
                    
                    connection_marker.lifetime = rospy.Duration(1.0)
                    marker_array.markers.append(connection_marker)
                    marker_id += 1
            
            # Publish the marker array
            if marker_array.markers:
                self.visualization_pub.publish(marker_array)
                rospy.loginfo("Published trajectory visualization with {} markers".format(len(marker_array.markers)))
            
        except Exception as e:
            rospy.logerr("Error publishing trajectory visualization: {}".format(str(e)))
    
    def create_prediction_table(self, predicted_points, current_position):
        """
        Create and display a table comparing ground truth, observations, and predictions
        """
        try:
            self.prediction_counter += 1
            current_time = rospy.Time.now().to_sec()
            
            # Prepare table data
            table_data = []
            headers = ["Frame", "GT_x", "GT_y", "Obs_x", "Obs_y", "Pred_x", "Pred_y"]
            
            # Add observed data (historical points)
            observed_points = list(self.trajectory_points)
            for i, (obs_x, obs_y) in enumerate(observed_points):
                frame_num = self.prediction_counter - len(observed_points) + i + 1
                if frame_num > 0:
                    # For observed points, GT and Obs are the same (assuming perfect detection)
                    table_data.append([
                        frame_num,
                        "{:.4f}".format(obs_x),
                        "{:.4f}".format(obs_y),
                        "{:.4f}".format(obs_x),
                        "{:.4f}".format(obs_y),
                        "-",
                        "-"
                    ])
            
            # Add predicted data (future points)
            for i, (pred_x, pred_y) in enumerate(predicted_points):
                frame_num = self.prediction_counter + i + 1
                table_data.append([
                    frame_num,
                    "-",  # Ground truth not available for future
                    "-",
                    "-",  # No observations for future
                    "-",
                    "{:.4f}".format(pred_x),
                    "{:.4f}".format(pred_y)
                ])
            
            # Print table header
            print("\n" + "="*80)
            print("TRAJECTORY PREDICTION TABLE - Prediction #{}".format(self.prediction_counter))
            print("Time: {:.3f}".format(current_time))
            print("Current Position: ({:.4f}, {:.4f})".format(current_position[0], current_position[1]))
            print("="*80)
            
            # Print table using simple formatting
            self.print_formatted_table(headers, table_data)
            
            # Print summary statistics
            if len(observed_points) > 1:
                # Calculate velocity from last two observed points
                last_two = observed_points[-2:]
                if len(last_two) == 2:
                    dx = last_two[1][0] - last_two[0][0]
                    dy = last_two[1][1] - last_two[0][1]
                    velocity = np.sqrt(dx*dx + dy*dy) / self.prediction_time_step
                    direction = np.arctan2(dy, dx) * 180 / np.pi
                    
                    print("\nSUMMARY:")
                    print("Observed velocity: {:.4f} m/s".format(velocity))
                    print("Direction: {:.1f} degrees".format(direction))
                    print("Prediction horizon: {:.1f} seconds".format(self.prediction_steps * self.prediction_time_step))
            
            print("="*80)
            
        except Exception as e:
            rospy.logerr("Error creating prediction table: {}".format(str(e)))
    
    def print_formatted_table(self, headers, data):
        """
        Print a nicely formatted table without external dependencies
        """
        try:
            # Calculate column widths
            col_widths = []
            for i, header in enumerate(headers):
                max_width = len(header)
                for row in data:
                    if i < len(row):
                        max_width = max(max_width, len(str(row[i])))
                col_widths.append(max_width + 2)  # Add padding
            
            # Print header
            header_line = "|"
            for i, header in enumerate(headers):
                header_line += " {:<{}} |".format(header, col_widths[i]-1)
            print(header_line)
            
            # Print separator
            separator = "|"
            for width in col_widths:
                separator += "-" * width + "|"
            print(separator)
            
            # Print data rows
            for row in data:
                data_line = "|"
                for i, cell in enumerate(row):
                    if i < len(col_widths):
                        data_line += " {:<{}} |".format(str(cell), col_widths[i]-1)
                print(data_line)
            
        except Exception as e:
            rospy.logerr("Error formatting table: {}".format(str(e)))

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