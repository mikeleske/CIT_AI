#!/usr/bin/env python3

import rospy
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import gym
from baselines import deepq

rospy.init_node('ur3_training_dqn', anonymous=True, log_level=rospy.WARN)


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    #is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    is_solved=False
    return is_solved


def main():
    task_and_robot_environment_name = rospy.get_param('/ur3_v0/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        #exploration_fraction=0.1,
        exploration_fraction=0.5,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to ur3_model.pkl")
    act.save("ur3_model.pkl")


if __name__ == '__main__':
    main()
