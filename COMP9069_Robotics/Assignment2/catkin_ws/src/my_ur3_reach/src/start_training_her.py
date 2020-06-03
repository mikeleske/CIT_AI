#!/usr/bin/env python3

import rospy
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import gym
from baselines.her import her

rospy.init_node('ur3_training_her', anonymous=True, log_level=rospy.WARN)


def main():
    #task_and_robot_environment_name = rospy.get_param('/ur3_v0/task_and_robot_environment_name')
    #env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    
    act = her.learn(
        env='UR3-v0',
        #env=env,
        network='mlp',
        total_timesteps=100000
    )
    print("Saving model to ur3_model.pkl")
    act.save("ur3_model.pkl")


if __name__ == '__main__':
    main()
