#!/usr/bin/env python

import rospy
import actionlib
import math
import numpy as np
from assignment1.msg import TurtleMoveFeedback, TurtleMoveResult, TurtleMoveAction
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose


class CollectionActionServer(object):

    # create messages that are used to publish feedback/result
    _feedback = TurtleMoveFeedback()
    _feedback.progress = '|..........|'
    _result = TurtleMoveResult()

    _TARGET_NODE = 'turtle1'
    _NODE_NAME = 'turtle1_collection_action_server'

    _ANGULAR_SPEED = np.pi/6 # set angular velocity to 30 degrees/sec
    _LINEAR_SPEED  = 0.5

    _TOLERANCE = 0.005
    _DISTANCE_TOLERANCE = 0.1

    def __init__(self):
        self._as = actionlib.SimpleActionServer(self._NODE_NAME, TurtleMoveAction, self.goal_callback, auto_start=False)
        self._as.start()
        rospy.loginfo("[{}::init] - CollectionActionServer launched successfully".format(self._NODE_NAME))
        self._rate = rospy.Rate(200) # 200Hz

        self._move_pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=1)
        self._move_msg = Twist()

        self._pose = rospy.Subscriber('/turtle1/pose', Pose, self.sub_callback)
        self._x = 0
        self._y = 0
        self._theta = 0

        self._success = True

    def sub_callback(self, msg):
        '''
        Handle Subscriber messages for /turtle1/pose
        Update local variables
        '''
        self._x = msg.x
        self._y = msg.y
        self._theta = msg.theta
    
    def update_feedback(self, distance, goal, clear=False):
        '''
        Publish updates on Feedback topic.
        Feedback is provided as reminaing time in seconds and progressbar.
        '''
        cur_distance = self.euclidean_distance(goal)
        delta = distance - cur_distance
        progress = int(round(delta/distance*10))
        progress_bar = '|{}|'.format("=" * progress + "." * (10 - progress))

        if clear:
            self._feedback.remaining_time = str(0.0)
            self._feedback.progress = '|==========|'
            self._as.publish_feedback(self._feedback)
        else:
            self._feedback.remaining_time = str(cur_distance/self._LINEAR_SPEED)
            self._feedback.progress = progress_bar
            self._as.publish_feedback(self._feedback)

    def publish_once(self, pub, msg):
        while True:
            connections = pub.get_num_connections()
            if connections > 0:
                pub.publish(msg)
                break
            else:
                self._rate.sleep()

    def turn_left(self):
        '''
        Instruct turtle1 to move left
        '''
        self._move_msg.linear.x = 0
        self._move_msg.angular.z = self._ANGULAR_SPEED
        self.publish_once(self._move_pub, self._move_msg)

    def turn_right(self):
        '''
        Instruct turtle1 to move right
        '''
        self._move_msg.linear.x = 0
        self._move_msg.angular.z = -self._ANGULAR_SPEED
        self.publish_once(self._move_pub, self._move_msg)

    def turn(self, angle):
        '''
        Function to calculate shortest direction to turn towards next object
        '''
        rospy.loginfo("[{}::turn] - Turning {}".format(self._NODE_NAME, self._TARGET_NODE))
        
        if (self._theta + np.pi) < (angle + np.pi):
            if abs((self._theta + np.pi) - (angle + np.pi)) < np.pi:
                while abs((self._theta + np.pi) - (angle + np.pi)) > self._TOLERANCE:
                    self.turn_left()
                    self._rate.sleep()
            else:
                while abs((self._theta + np.pi) - (angle + np.pi)) > self._TOLERANCE:
                    self.turn_right()
                    self._rate.sleep()
        else:
            if abs((self._theta + np.pi) - (angle + np.pi)) < np.pi:
                while abs((self._theta + np.pi) - (angle + np.pi)) > self._TOLERANCE:
                    self.turn_right()
                    self._rate.sleep()
            else:
                while abs((self._theta + np.pi) - (angle + np.pi)) > self._TOLERANCE:
                    self.turn_left()
                    self._rate.sleep()
        self.stop()

        rospy.loginfo("[{}::turn] - Turning {} completed".format(self._NODE_NAME, self._TARGET_NODE))

    def euclidean_distance(self, goal):
        '''
        Calculate distance between turtle1 and next object
        '''
        return math.sqrt((goal.x - self._x)**2 + (goal.y - self._y)**2)

    def move_straight(self, goal):
        '''
        Move turtle1 to next object
        '''
        self._move_msg.linear.x = self._LINEAR_SPEED
        self._move_msg.angular.z = 0

        rospy.loginfo("[{}::move_straight] - Moving {}".format(self._NODE_NAME, self._TARGET_NODE))
        
        distance = self.euclidean_distance(goal)

        counter = 0

        #
        # While turtle1 is moving, update the Feedback topic
        # Feedback update us limited to 2 updates per second
        #
        while self.euclidean_distance(goal) >= self._DISTANCE_TOLERANCE:
            if self._as.is_preempt_requested():
                rospy.loginfo('The goal has been cancelled/preempted')
                self._as.set_preempted()
                self._success = False
                break

            # Throttle feedback
            if counter % 100 == 0:
                self.update_feedback(distance, goal)
            
            self.publish_once(self._move_pub, self._move_msg)

            counter += 1
            self._rate.sleep()
        self.stop()

        self.update_feedback(distance, goal, clear=True)

        rospy.loginfo("[{}::move_straight] - Moving {} completed".format(self._NODE_NAME, self._TARGET_NODE))

    def stop(self):
        '''
        Stop all turtle1 movements
        '''
        self._move_msg.linear.x = 0
        self._move_msg.angular.z = 0
        self.publish_once(self._move_pub, self._move_msg)

    def angle_between_points(self, p1, p2):
        '''
        Calculate angle between turtle1 theta and target
        '''
        radians = math.atan2(p1[1]-p2[1], p1[0]-p2[0])
        return radians

    def goal_callback(self, goal):
        '''
        Handle Goal Request from client with next coordinates
        Calculate angle to turn.
        Turn turtle1
        Move turtle1 to target
        '''
        rospy.loginfo("[{}::goal_callback] - Executing goal callback function".format(self._NODE_NAME))

        self._success = True
        
        rospy.loginfo("[{}::goal_callback] - {} next object location {} {}".format(
            self._NODE_NAME, self._TARGET_NODE, goal.x, goal.y))
        
        angle = self.angle_between_points((goal.x, goal.y), (self._x, self._y))

        # Turn turtle and move it
        self.turn(angle)
        self.move_straight(goal)

        # Check success
        if self._success:
            self._result.success = True
            self._as.set_succeeded(self._result)
        
        # Return result
        return self._result


if __name__ == '__main__':
    rospy.init_node('turtle1_collection_action_server')
    CollectionActionServer()
    rospy.spin()
