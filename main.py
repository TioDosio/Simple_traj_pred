
from __future__ import division, print_function
import rospy
import numpy as np
from collections import deque
from geometry_msgs.msg import PointStamped, Point
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class SimpleTrajectoryPredictor:
    def __init__(self):
        rospy.init_node('simple_trajectory_predictor', anonymous=True)
        
        # Parameters
        self.window_size = 10  # Number of points to keep for prediction
        self.prediction_steps = 5  # Number of future steps to predict
        self.prediction_time_step = 0.1  # Time step for predictions (seconds)
        
        # Data storage
        self.trajectory_points = deque(maxlen=self.window_size)
        self.timestamps = deque(maxlen=self.window_size)
        
        # Subscribers and Publishers
        self.image_detections_sub = rospy.Subscriber('/person_coordinates', PointStamped, self.image_detections_callback)
        self.prediction_pub = rospy.Publisher('/predicted_trajectory', PoseArray, queue_size=10)
        
        rospy.loginfo("Simple Trajectory Predictor initialized")
    
    def image_detections_callback(self, msg):
        """
        Callback function for receiving person coordinates
        """
        try:
            # Extract coordinates and timestamp
            x = msg.point.x
            y = msg.point.y
            timestamp = msg.header.stamp.to_sec()
            
            # Store the point and timestamp
            self.trajectory_points.append((x, y))
            self.timestamps.append(timestamp)
            
            rospy.loginfo("Received point: ({:.2f}, {:.2f}) at time {:.3f}".format(x, y, timestamp))
            
            # Predict trajectory if we have enough points
            if len(self.trajectory_points) >= 3:
                predicted_trajectory = self.predict_trajectory()
                self.publish_prediction(predicted_trajectory)
                
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
            pose_array.header.frame_id = "map"  # or appropriate frame
            
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
    
    def get_trajectory_velocity(self):
        """
        Calculate current velocity based on recent points
        """
        if len(self.trajectory_points) < 2:
            return 0.0, 0.0
        
        points = np.array(list(self.trajectory_points))
        times = np.array(list(self.timestamps))
        
        # Calculate velocity from last two points
        dx = points[-1][0] - points[-2][0]
        dy = points[-1][1] - points[-2][1]
        dt = times[-1] - times[-2]
        
        if dt > 0:
            vx = dx / dt
            vy = dy / dt
            return vx, vy
        return 0.0, 0.0
    
    def visualize_trajectory(self):
        """
        Simple visualization of current trajectory and prediction
        """
        if len(self.trajectory_points) < 2:
            return
        
        try:
            points = np.array(list(self.trajectory_points))
            predicted = self.predict_trajectory()
            
            plt.figure(figsize=(10, 6))
            
            # Plot historical points
            plt.plot(points[:, 0], points[:, 1], 'bo-', label='Historical Points', markersize=6)
            
            # Plot predicted points
            if predicted:
                pred_points = np.array(predicted)
                plt.plot(pred_points[:, 0], pred_points[:, 1], 'ro--', label='Predicted Points', markersize=6)
            
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.title('Trajectory Prediction')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.show()
            
        except Exception as e:
            rospy.logerr("Error in visualization: {}".format(str(e)))

def main():
    """
    Main function to run the trajectory predictor
    """
    try:
        predictor = SimpleTrajectoryPredictor()
        rospy.loginfo("Trajectory predictor is running...")
        rospy.loginfo("Waiting for person coordinates on /person_coordinates topic...")
        
        # Keep the node running
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Trajectory predictor node interrupted")
    except Exception as e:
        rospy.logerr("Error in main: {}".format(str(e)))

if __name__ == "__main__":
    main()