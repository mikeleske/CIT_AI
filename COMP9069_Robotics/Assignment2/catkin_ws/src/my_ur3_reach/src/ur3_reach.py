#!/usr/bin/env python

import time
import rospy
import tf

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from control_msgs.msg import JointControllerState

JOINTS = []
PUBS   = []

rospy.init_node('ur3_reach')
rate = rospy.Rate(10)

listener = tf.TransformListener()

def js_callback(msg):
    global JOINTS
    JOINTS = msg

def create_pubs():
    '''
    Create all ROS Joint Publishers needed.
    '''
    global JOINTS
    global PUBS

    for j in JOINTS.name:
        path = '/ur3/' + j + '_position_controller/command'
        pub = rospy.Publisher(path, Float64, queue_size=1)
        PUBS.append(pub)

def get_input():
    print('\n\nAvailable joints:')

    for i, joint in enumerate(JOINTS.name):
        print('  {}  {}'.format(i, joint))
    
    print('  {}  {}'.format('9', 'Exit'))
    i = int(input("Select a joint to move: "))
    if i == 9:
        return i, 0
    p = float(input("Set position for joint: "))

    return i, p


def move_joint(i, p):
    '''
    Move a given joint to its position
    '''
    global PUBS
    global JOINTS
    global listener

    pub = PUBS[i]
    msg = Float64()
    msg.data = p

    publish_once(pub, msg)

    #
    # Verify if joint reached set position
    #
    REACHED_POS = False
    for _ in range(20):
        rospy.sleep(0.1)

        if abs(JOINTS.position[i] - p) <= 0.01:
            REACHED_POS = True
            break
    
    if not REACHED_POS:
        rospy.logerr("ERROR: Robot arm did not reach target position.")


def print_output(i, p):

    global JOINTS
    global listener

    act_pos = JOINTS.position[i]
    error   = JOINTS.position[i] - p
    (pos, rot) = listener.lookupTransform("base_link", "wrist_3_link", rospy.Time(0))

    print('\nJoint {} actual position: {}'.format(JOINTS.name[i], act_pos))
    print('Joint {} error          : {}'.format(JOINTS.name[i], error))
    print('End-Effector to base position    [x, y, z]     : {}'.format(pos))
    print('End-Effector to base orientation [x, y, z, w]: {}'.format(rot))


def publish_once(pub, msg):
    while True:
        connections = pub.get_num_connections()
        if connections > 0:
            pub.publish(msg)
            break
        else:
            rate.sleep()

if __name__ == "__main__":
    
    # Subscribe to /ur3/joint_states to learn about available joints
    js = rospy.Subscriber('/ur3/joint_states', JointState, js_callback)
    time.sleep(1)

    # Create ROS publishers
    create_pubs()

    # Let the user play with the robot
    while True:
        i, p = get_input()
        
        if i == 9:
            exit('Exiting.')
        else:
            move_joint(i, p)
            print_output(i, p)


