#!/usr/bin/env python

import rospy
import random
import time
from assignment1.srv import SpawnObjectsMessage, SpawnObjectsMessageResponse 
from assignment1.srv import SpawnedObjectMessage, SpawnedObjectMessageRequest
from turtlesim.srv import Spawn, SpawnRequest, SpawnResponse


class SpawnObjectsServer():

    _NODE_NAME = 'spawn_objects_service_server'
    _MOUNT = '/spawn_objects'

    def __init__(self, name):
        self._NODE_NAME = name
        rospy.loginfo("[{}] - SpawnObjectsServer launched".format(self._NODE_NAME))

        _ = rospy.Service(self._MOUNT, SpawnObjectsMessage , self.service_callback)
        rospy.loginfo("[{}::init] - Service ready at {}".format(self._NODE_NAME, self._MOUNT))

        rospy.wait_for_service('/spawned_object')
        self.spawned_obj_srv = rospy.ServiceProxy('/spawned_object', SpawnedObjectMessage)

        self.ctrl_c = False
        rospy.on_shutdown(self.shutdownhook)

    def shutdownhook(self):
        # works better than rospy.is_shutdown()
        self.ctrl_c = True

    def spawn(self, count):
        '''
        Spwan objects by calling turtlesim /spawn service
        '''
        rospy.wait_for_service('/spawn')
        rospy.loginfo("[{}::spawn] - Service /spawn ready".format(self._NODE_NAME))
        rospy.loginfo("[{}::spawn] - Spawning {} objects".format(self._NODE_NAME, count))

        service = rospy.ServiceProxy('/spawn', Spawn)

        for i in range(count):

            # Get random coordinates and set name
            x = random.randint(0, 10)
            y = random.randint(4, 10)
            name = 'object' + str(i)
            
            # Create SpawnRequest message and set parameters
            srv_msg = SpawnRequest()
            srv_msg.x = x
            srv_msg.y = y
            srv_msg.name = name

            # Send request
            result = service.call(srv_msg)

            # Update NextObjectServer with newly spawned object
            self.update_object_server(name, x, y)

            rospy.loginfo("[{}::spawn] - Spawned {} at x:{} y:{}".format(self._NODE_NAME, name, x, y))

    def update_object_server(self, name, x, y):
        '''
        Send a Service Message to NextObjectServer to make server aware of new object
        '''
        req = SpawnedObjectMessageRequest()
        req.name = name
        req.x = x
        req.y = y

        _ = self.spawned_obj_srv.call(req)

    def service_callback(self, request):
        '''
        Service callback for object spawner.
        Request received contains number of objects to be spawned
        '''
        rospy.loginfo("[{}::service_callback] - Service has been called".format(self._NODE_NAME))
        
        # spawn objects
        self.spawn(request.count)
        rospy.loginfo("[{}::service_callback] - Finished service".format(self._NODE_NAME))
        
        # Prepare and send response
        response = SpawnObjectsMessageResponse()
        response.response = True
        return response


if __name__ == '__main__':
   rospy.init_node('spawn_objects_service_server', anonymous=True)
   sos = SpawnObjectsServer('spawn_objects_service_server')
   time.sleep(2)
   try:
      rospy.spin()
   except rospy.ROSInterruptException:
      pass