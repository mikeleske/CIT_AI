#!/usr/bin/env python

import rospy
import actionlib

import time
from turtlesim.msg import Pose
from turtlesim.srv import Kill, KillRequest
from assignment1.msg import TurtleMoveGoal, TurtleMoveFeedback, TurtleMoveResult, TurtleMoveAction
from assignment1.srv import NextObjectMessage, NextObjectMessageRequest, NextObjectMessageResponse


class Turtle1Client(object):

    def __init__(self, name):
        self._NODE_NAME = name
        rospy.loginfo("[{}::init] - Turtle1Client launched successfully".format(self._NODE_NAME))

        self._rate = rospy.Rate(10)

        self._pose = rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)
        self._x = 0
        self._y = 0
        self._theta = 0

        self._init_x = 0
        self._init_y = 0

        self.move_client = actionlib.SimpleActionClient('/turtle1_collection_action_server', TurtleMoveAction)
        rospy.loginfo("[{}::init] - Waiting for Action Server /turtle1_collection_action_server".format(self._NODE_NAME))
        self.move_client.wait_for_server()
        rospy.loginfo("[{}::init] - Action Server /turtle1_collection_action_server found".format(self._NODE_NAME))

    def pose_callback(self, msg):
        '''
        Handle Subscriber messages for /turtle1/pose
        Update local variables
        '''
        self._x = msg.x
        self._y = msg.y
        self._theta = msg.theta

    def next_object(self):
        '''
        Call NextObjectServer to learn which object to collect next
        Tell NextObjectServer current location
        Return target coordinates
        '''
        rospy.loginfo("[{}::next_object] - Requesting next object from object server".format(self._NODE_NAME))
        rospy.wait_for_service('/next_object')
        service = rospy.ServiceProxy('/next_object', NextObjectMessage)
        rospy.loginfo("[{}::next_object] - Connected to service /next_object".format(self._NODE_NAME))

        req = NextObjectMessageRequest()
        req.x = self._x
        req.y = self._y

        rospy.loginfo("[{}::next_object] - Sending service request next object".format(self._NODE_NAME))
        result = service(req)
        rospy.loginfo("[{}::next_object] - Got next object: x {}, y {}, name {}".format(self._NODE_NAME, result.x, result.y, result.name))

        return result
    
    def feedback_callback(self, feedback):
        pass

    def go_to_object(self, x, y):
        '''
        Create a goal object to send to the action server and set the attribute values
        Goal object defines coordinate turtle1 want to be directed
        '''
        goal = TurtleMoveGoal()
        goal.x = x
        goal.y = y

        # Send goal to action server and wait for result
        self.move_client.send_goal(goal, feedback_cb=self.feedback_callback)
        self.move_client.wait_for_result()
        result = self.move_client.get_result()

        return result.success

    def delete_object(self, name):
        '''
        Call /kick service to delete collected object
        '''
        rospy.wait_for_service('/kill')
        service = rospy.ServiceProxy('/kill', Kill)
        req = KillRequest()
        req.name = name
        _ = service(req)

    def run(self):
        '''
        Main loop for turtle1 client
        '''
        rospy.loginfo("[{}::run] - turtle1 starts collecting objects".format(self._NODE_NAME))

        # Once started, memorize initial position
        self._init_x = self._x
        self._init_y = self._y

        while True:
            # Ask NextObjectServer for next object to collect
            next_object = self.next_object()
            success = self.go_to_object(next_object.x, next_object.y)

            # If the object was reached, delete object
            if success:
                self.delete_object(next_object.name)
            
            # If object was last remaining object, break look
            if next_object.last == 1:
                rospy.loginfo("[{}::run] - All objects collected".format(self._NODE_NAME))
                break
        
        # Return turtle1 to init position
        rospy.loginfo("[{}::run] - Send turtle1 to init position".format(self._NODE_NAME))
        _ = self.go_to_object(self._init_x, self._init_y)


if __name__ == '__main__':
    rospy.init_node('turtle1_client')
    client = Turtle1Client('turtle1_client')
    
    # TODO: Find better way to wait for initialization
    time.sleep(5)
    client.run()