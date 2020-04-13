#!/usr/bin/env python

import rospy
import actionlib

import time
import math
from turtlesim.msg import Pose
from turtlesim.srv import Kill, KillRequest
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from assignment1.msg import TurtleMoveGoal, TurtleMoveFeedback, TurtleMoveResult, TurtleMoveAction
from assignment1.srv import NextObjectMessage, NextObjectMessageRequest, NextObjectMessageResponse


class Turtle2Client(object):

    _PREFIX = 'conveyor_obj'
    _LINEAR_SPEED  = 1.0
    _DISTANCE_TOLERANCE = 0.25

    def __init__(self, name):
        self._NODE_NAME = name
        rospy.loginfo("[{}::init] - Turtle2Client launched successfully".format(self._NODE_NAME))

    	# Set the rospy rate to 10Hz
        self._rate = rospy.Rate(10)

        # This Topic Subscriber reads turtle2's own pose
        self._pose = rospy.Subscriber('/turtle2/pose', Pose, self.pose_callback)
        self._x = 0
        self._y = 0
        self._theta = 0

        # This Topic Publisher handles turtle2's movements
        self._move_pub = rospy.Publisher('/turtle2/cmd_vel', Twist, queue_size=1)
        self._move_msg = Twist()

        # This Topic Publisher updates ConveyorBelt to kill objects
        self._remove_pub = rospy.Publisher('/conveyor_belt/remove', String, queue_size=1)
        self._remove_msg = String()

        # This Topic Publisher updates ConveyorBelt that Turtle2 is ready to collect
        self._produce_pub = rospy.Publisher('/conveyor_belt/produce', Bool, queue_size=1)
        self._produce_msg = Bool()

        # This Topic Subscriber listens to ConveyorBelt spawned objects
        self._spawned_sub = rospy.Subscriber('/conveyor_belt/spawned', String, self.spawned_callback)

        # Track the current objects on the conveyor belt
        self._spawned_objs = []

        # Define the y positions of interest
        self._y_wait = 3
        self._y_collect = 1

        rospy.on_shutdown(self.shutdownhook)

    def shutdownhook(self):
        self.update_produce(False)
        self.ctrl_c = True

    def spawned_callback(self, msg):
        '''
        Add new objects to tracking list: self._spawned_objs
        '''
        name = msg.data
        rospy.loginfo("[{}::spawned_callback] - Turtle2Client learned about new onbject {}".format(self._NODE_NAME, name))
        self._spawned_objs.append(name)

    def publish_once(self, pub, msg):
        while True:
            connections = pub.get_num_connections()
            if connections > 0:
                pub.publish(msg)
                break
            else:
                self._rate.sleep()

    def pose_callback(self, msg):
        '''
        Update turtle pose
        '''
        self._x = msg.x
        self._y = msg.y
        self._theta = msg.theta
    
    def euclidean_distance(self, goal):
        '''
        Calculate distance between turtle2 and approaching object on conveyor belt
        '''
        return math.sqrt((goal.x - self._x)**2 + (goal.y - self._y)**2)

    def verify_distance(self, obj):
        '''
        Verify a given object is within a certain tolerance radius
        '''
        path = '/{}/pose'.format(obj)
        pose = rospy.wait_for_message(path, Pose)
        
        distance = self.euclidean_distance(pose)

        return distance < self._DISTANCE_TOLERANCE

    def stop(self):
        '''
        Halt turtle2
        '''
        self._move_msg.linear.x = 0
        self._move_msg.angular.z = 0
        self.publish_once(self._move_pub, self._move_msg)

    def move_down(self):
        '''
        Move turtle2 down to belt position
        '''
        rospy.loginfo("[{}::move_down] - Turtle2Client moving down".format(self._NODE_NAME))
        self._move_msg.linear.x = self._LINEAR_SPEED
        self._move_msg.angular.z = 0

        while self._y > self._y_collect:
            self.publish_once(self._move_pub, self._move_msg)
            self._rate.sleep()
        self.stop()
    
    def move_up(self):
        '''
        Move turtle2 up to wait position
        '''
        rospy.loginfo("[{}::move_up] - Turtle2Client moving back".format(self._NODE_NAME))
        self._move_msg.linear.x = -self._LINEAR_SPEED
        self._move_msg.angular.z = 0

        while self._y < self._y_wait:
            self.publish_once(self._move_pub, self._move_msg)
            self._rate.sleep()
        self.stop()
    
    def remove(self, obj):
        '''
        Remove a turtle from the belt. This involves publushing the object in question to '/conveyor_belt/remove'
        '''
        rospy.loginfo("[{}::remove] - Turtle2Client removing {}".format(self._NODE_NAME, obj))
        
        if self.verify_distance(obj):
            rospy.loginfo("[{}::remove] - Distance check successful".format(self._NODE_NAME))
            self._remove_msg.data = obj
            self.publish_once(self._remove_pub, self._remove_msg)
            self._spawned_objs.remove(obj)
        else:
            rospy.loginfo("[{}::remove] - Distance check not successful - Missed object".format(self._NODE_NAME))

    def check_objects(self):
        '''
        Check all conveyor belt objects' pose.
        If pose in a certain x range, initiate turtle actions: move down, remove, move up
        '''
        for obj in self._spawned_objs:
            # Read the pose of the object
            path = '/{}/pose'.format(obj)
            pose = rospy.wait_for_message(path, Pose)

            if 3.3 < pose.x < 3.5:
                self.move_down()
                self.remove(obj)
                self.move_up()
    
    def update_produce(self, state):
        '''
        Update ConveyorBelt /conveyor_belt/produce with new state
        '''
        rospy.loginfo("[{}::update_produce] - Update ConveyorBelt: {}".format(self._NODE_NAME, state))
        self._produce_msg.data = state
        self.publish_once(self._produce_pub, self._produce_msg)

    def run(self):
        '''
        Main turtle2 loop - He's a star and not getting tired.
        '''

        # Tell the ConveyorBelt that turtle2 is ready to collect.
        self.update_produce(True)

        while True:
            if self._spawned_objs:
                self.check_objects()

            self._rate.sleep()

if __name__ == '__main__':
    rospy.init_node('turtle2_client')
    client = Turtle2Client('turtle2_client')
    
    # TODO: Find better way to wait for initialization
    time.sleep(2)
    client.run()