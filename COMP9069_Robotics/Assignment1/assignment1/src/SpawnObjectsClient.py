#! /usr/bin/env python

import rospy
import math
import random
import time
from assignment1.srv import SpawnObjectsMessage, SpawnObjectsMessageRequest

_NODE_NAME = 'spawn_objects_client'


rospy.init_node(_NODE_NAME)
rospy.loginfo("[{}::main] - Client running".format(_NODE_NAME))

rospy.wait_for_service('/spawn_objects')
spawn = rospy.ServiceProxy('/spawn_objects', SpawnObjectsMessage)
rospy.loginfo("[{}::main] - Connected to service /spawn_objects".format(_NODE_NAME))


# Give server time to fully init
time.sleep(2)

# Create SpawnObjectsMessageRequest object
# Get random number between 5 and 10
req = SpawnObjectsMessageRequest()
req.count = random.randint(5, 10)

rospy.loginfo("[{}::main] - Sending service request to spawn {} objects".format(_NODE_NAME, req.count))
result = spawn(req)

if result:
    rospy.loginfo("[{}::main] - Objects successfully spawned".format(_NODE_NAME))
else:
    rospy.logerr("[{}::main] - An error occurred during object creation".format(_NODE_NAME))

rospy.loginfo("[{}::main] - Shutting down".format(_NODE_NAME))
