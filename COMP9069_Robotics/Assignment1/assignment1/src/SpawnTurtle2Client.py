#! /usr/bin/env python

import rospy
import math
from turtlesim.srv import Spawn, SpawnRequest

_NODE_NAME = 'spawn_turtle2_client'


rospy.init_node(_NODE_NAME)
rospy.loginfo("[{}::main] - Client running".format(_NODE_NAME))

rospy.wait_for_service('/spawn')
spawn = rospy.ServiceProxy('/spawn', Spawn)
rospy.loginfo("[{}::main] - Connected to service /spawn".format(_NODE_NAME))

#
# Spawn turtle2 in given locaton and orientation
#
req = SpawnRequest()
req.x = 5.5
req.y = 3
req.theta = - math.pi / 2
req.name = 'turtle2'

rospy.loginfo("[{}::main] - Sending service request to spawn turtle2".format(_NODE_NAME))
result = spawn(req)

if result:
    rospy.loginfo("[{}::main] - {} successfully spawned".format(_NODE_NAME, result.name))

rospy.loginfo("[{}::main] - Shutting down".format(_NODE_NAME))
