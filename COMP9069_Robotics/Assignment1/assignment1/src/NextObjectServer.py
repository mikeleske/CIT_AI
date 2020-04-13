#!/usr/bin/env python

import rospy
import math
from turtlesim.msg import Pose
from assignment1.srv import NextObjectMessage, NextObjectMessageResponse 
from assignment1.srv import SpawnedObjectMessage, SpawnedObjectMessageResponse

class NextObjectServer():

    _MOUNT_NO  = '/next_object'
    _MOUNT_SO = '/spawned_object'
    _PREFIX = 'object'

    def __init__(self, name):
        self._NODE_NAME = name
        rospy.loginfo("[{}::init] - NextObjectServer launched".format(self._NODE_NAME))

        _ = rospy.Service(self._MOUNT_NO, NextObjectMessage , self.next_object_callback)
        rospy.loginfo("[{}::init] - Service ready at {}".format(self._NODE_NAME, self._MOUNT_NO))

        _ = rospy.Service(self._MOUNT_SO, SpawnedObjectMessage , self.spawned_object_callback)
        rospy.loginfo("[{}::init] - Service ready at {}".format(self._NODE_NAME, self._MOUNT_SO))

        self._TURTLE_X = 0
        self._TURTLE_Y = 0

        self._objects = {}

        self.ctrl_c = False
        rospy.on_shutdown(self.shutdownhook)

    def shutdownhook(self):
        self.ctrl_c = True

    def next_object_callback(self, request):
        '''
        Handle service callbacks for /next_object from turtle1
        '''
        rospy.loginfo("[{}::next_object_callback] - Service has been called".format(self._NODE_NAME))
        
        # Process service message and update local variables (turtle1 coordinates)
        self._TURTLE_X = request.x
        self._TURTLE_Y = request.y

        # Select the next object
        next_object = self.select()
        
        # Prepare response message with next object information
        response = NextObjectMessageResponse()
        response.x = next_object[0]
        response.y = next_object[1]
        response.name = next_object[2]
        response.last = next_object[3]

        rospy.loginfo("[{}] - Finished service".format(self._NODE_NAME))

        return response

    def spawned_object_callback(self, request):
        '''
        Handle service callbacks for /spawned_object from SpawnObjectsServer
        '''
        rospy.loginfo("[{}::spawned_object_callback] - Service has been called".format(self._NODE_NAME))
        
        # Add new object to local data structure for tracking _objects
        self._objects[request.name] = {}
        self._objects[request.name]['x'] = request.x
        self._objects[request.name]['y'] = request.y
        
        # Create response message
        response = SpawnedObjectMessageResponse()
        response.status = True

        rospy.loginfo("[{}] - Finished service".format(self._NODE_NAME))

        return response

    def select(self):
        '''
        Get nearest object.
        This method implements a greedy approach to return the closest next object.
        '''
        next_object = None

        rospy.loginfo("[{}::select] - NextObjectServer - Found {} remaining objects".format(self._NODE_NAME, str(len(self._objects))))
        
        # Iterate over objects known to NextObjectServer
        for obj, data in self._objects.items():
            # Calculate eucleadian distance between turtle1 and objects
            distance = math.sqrt((self._TURTLE_X - data['x'])**2 + (self._TURTLE_Y - data['y'])**2)
            if not next_object or distance < next_object[4]:
                next_object = (data['x'], data['y'], obj, len(self._objects)==1, distance)
        
        # Delete selected object from local data structure
        del self._objects[next_object[2]]

        rospy.loginfo("[{}::select] - NextObjectServer - Next object: {}".format(self._NODE_NAME, next_object))

        # Return next object
        return next_object

if __name__ == '__main__':
   rospy.init_node('next_object_service_server', anonymous=True)
   nos = NextObjectServer('next_object_service_server')
   try:
      rospy.spin()
   except rospy.ROSInterruptException:
      pass