#!/usr/bin/env python

import rospy
import random
import time

from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from turtlesim.srv import Kill, KillRequest
from turtlesim.srv import Spawn, SpawnRequest, SpawnResponse

class ConveyorBelt():

    _LINEAR_SPEED = 1.0

    def __init__(self, name):
        self._NODE_NAME = name
        rospy.loginfo("[{}] - ConveyorBelt launched".format(self._NODE_NAME))

        # Set the rospy rate to 2Hz
        self._rate = rospy.Rate(2)

        self._produce = False

        self._objects = []
        self._publishers = {}
        self._spawn_count = 0

        # This Topic Publisher updates informs about spawned objects
        self._spawned_pub = rospy.Publisher('/conveyor_belt/spawned', String, queue_size=1)
        self._spawned_msg = String()

        # This Topic Subscriber listens to which object should be deleted
        self._remove = rospy.Subscriber('/conveyor_belt/remove', String, self.remove_callback)

        # This Topic Subscriber listens for an active collector/picker
        self._remove = rospy.Subscriber('/conveyor_belt/produce', Bool, self.produce_callback)

        self.ctrl_c = False
        rospy.on_shutdown(self.shutdownhook)

    def shutdownhook(self):
        # works better than rospy.is_shutdown()
        self.ctrl_c = True

    def publish_once(self, pub, msg):
        while True:
            connections = pub.get_num_connections()
            if connections > 0:
                pub.publish(msg)
                break
            else:
                self._rate.sleep()
    
    def remove_callback(self, msg):
        '''
        Callback function called, when /conveyor_belt/remove Subscriber sees new message
        '''
        # unregister Publisher to move object
        self._publishers[msg.data].unregister()
        del self._publishers[msg.data]

        # Kill the turtle object
        rospy.wait_for_service('/kill')
        rospy.loginfo("[{}::remove_callback] - ConveyorBelt - Service /kill ready".format(self._NODE_NAME))
        service = rospy.ServiceProxy('/kill', Kill)
        req = KillRequest()
        req.name = msg.data
        _ = service(req)
        self._objects.remove(msg.data)
        rospy.loginfo("[{}::remove_callback] - ConveyorBelt - Object killed".format(self._NODE_NAME))

 
    def produce_callback(self, msg):
        # Update produce flag with state received from turtle2
        self._produce = msg.data
        rospy.loginfo("[{}::produce_callback] - ConveyorBelt - Setting produce flag to {}".format(self._NODE_NAME, self._produce))

    def spawn(self):
        '''
        Spawn a new conveyor belt object
        '''
        rospy.wait_for_service('/spawn')
        rospy.loginfo("[{}::spawn] - ConveyorBelt - Service /spawn ready".format(self._NODE_NAME))

        service = rospy.ServiceProxy('/spawn', Spawn)
            
        srv_msg = SpawnRequest()
        srv_msg.x = 0
        srv_msg.y = 1
        srv_msg.name = 'conveyor_obj' + str(self._spawn_count)

        result = service.call(srv_msg)

        self._spawned_msg.data = srv_msg.name
        self.publish_once(self._spawned_pub, self._spawned_msg)

        rospy.loginfo("[{}::spawn] - ConveyorBelt - Object spawned".format(self._NODE_NAME))

        # Create a topic Publisher to move turtle
        pub = rospy.Publisher('/{}/cmd_vel'.format(srv_msg.name), Twist, queue_size=1)
        self._publishers[srv_msg.name] = pub

        return result

    def kick(self):
        '''
        Update each active objects and keep it moving
        '''
        for _, pub in self._publishers.items():
            msg = Twist()
            msg.linear.x = self._LINEAR_SPEED
            self.publish_once(pub, msg)


    def run(self):
        '''
        Main conveyor belt loop - It keeps running as long as there is electricity!
        '''
        rospy.loginfo("[{}] - ConveyorBelt is serving".format(self._NODE_NAME))

        i = 0

        # Set a random value to ensure new objects every 5-8 seconds 
        next_spawn = random.randint(10, 16)

        while not rospy.is_shutdown():
            if self._produce:
                self.kick()

                # Spawn new object when the next spawn step has been reached
                if i == next_spawn:
                    obj = self.spawn()
                    self._spawn_count += 1
                    self._objects.append(obj.name)

                    # reset i and set new next_spawn target
                    i = 0
                    next_spawn = random.randint(10, 16)

                i += 1
            
            self._rate.sleep()


if __name__ == '__main__':
   rospy.init_node('conveyor_belt', anonymous=True)
   cb = ConveyorBelt('conveyor_belt')
   try:
      cb.run()
   except rospy.ROSInterruptException:
      pass
