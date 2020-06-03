#!/usr/bin/env python

from gym import utils
from openai_ros.robot_envs import ur3_env
from gym.envs.registration import register
from gym import error, spaces
import rospy
import numpy as np
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from openai_ros.msg import RLExperimentInfo

import os

class UR3PosEnv(ur3_env.UR3Env):
    def __init__(self):

        ros_ws_abspath = rospy.get_param("/ur3_v0/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="my_ur3_description",
                    launch_file_name="start_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/ur3/config",
                               yaml_file_name="ur3_position_task.yaml")

        self.get_params()

        self.action_space = spaces.Discrete(self.n_actions)

        low = np.array([-np.pi, -np.pi, -np.pi])
        high = np.array([np.pi, np.pi, np.pi])
        self.observation_space = spaces.Box(low, high)

        self.tolerance = 0.2

        self.iteration = 0

        # Helps if last step in episode reaches goal
        self.reached_goal = False
        
        # For a little visualization
        self.pub_step_reward = rospy.Publisher('/ur3/step_reward', RLExperimentInfo, queue_size=1)
        self.total_iteration = 0

        # Needed for HER
        self.num_envs = 1

        self.rate = rospy.Rate(1)
        self.reset_timer = rospy.Rate(0.5)

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(UR3PosEnv, self).__init__(ros_ws_abspath=ros_ws_abspath)

    def get_params(self):

        # get configuration parameters
        self.n_actions = rospy.get_param('/ur3/n_actions')
        self.n_observations = rospy.get_param('/ur3/n_observations')
        self.max_iterations = rospy.get_param('/ur3/max_iterations')
        

        self.init_elbow = rospy.get_param('/ur3/init_pos/elbow_joint')
        self.init_shoulder_lift = rospy.get_param('/ur3/init_pos/shoulder_lift_joint')
        self.init_shoulder_pan = rospy.get_param('/ur3/init_pos/shoulder_pan_joint')
        self.init_wrist_1 = rospy.get_param('/ur3/init_pos/wrist_1_joint')
        self.init_wrist_2 = rospy.get_param('/ur3/init_pos/wrist_2_joint')
        self.init_wrist_3 = rospy.get_param('/ur3/init_pos/wrist_3_joint')
        self.init_pos = [self.init_elbow, self.init_shoulder_lift, self.init_shoulder_pan,
                         self.init_wrist_1, self.init_wrist_2, self.init_wrist_3]

        self.goal_shoulder_pan = rospy.get_param('/ur3/goal_pos/shoulder_pan_joint')
        self.goal_shoulder_lift = rospy.get_param('/ur3/goal_pos/shoulder_lift_joint')
        self.goal_elbow = rospy.get_param('/ur3/goal_pos/elbow_joint')
        self.goal_pos = [self.goal_elbow, self.goal_shoulder_lift, self.goal_shoulder_pan]

        self.position_delta = rospy.get_param('/ur3/position_delta')
        self.reached_goal_reward = rospy.get_param('/ur3/reached_goal_reward')

        self.position_delta = rospy.get_param('/ur3/position_delta')

        self.state_dicretization = rospy.get_param('/ur3_v0/state_dicretization')

    def _set_action(self, action):

        # Take action
        #
        # self.pos[0] -> elbow
        # self.pos[1] -> shoulder_lift
        # self.pos[2] -> shoulder_pan
        #

        if action == 0:  # elbow -
            self.pos[0] -= self.position_delta
        elif action == 1:  # elbow +
            self.pos[0] += self.position_delta
        elif action == 2:  # shoulder_lift -
            self.pos[1] -= self.position_delta
        elif action == 3:  # shoulder_lift -
            self.pos[1] += self.position_delta
        elif action == 4:  # shoulder_pan -
            self.pos[2] -= self.position_delta
        elif action == 5:  # shoulder_pan -
            self.pos[2] += self.position_delta

        # Move the joints
        self.move_joints(self.pos)

    def _get_obs(self):

        obs = [ round(i, self.state_dicretization) for i in self.joints[0:3] ]
        return np.array(obs)

    def _is_done(self, observations):
        done = False

        # check if 3 joints of interest are in goal tolerance
        if ((abs(self.goal_elbow - observations[0]) <= self.tolerance) and 
            abs(self.goal_shoulder_lift - observations[1]) <= self.tolerance and 
            abs(self.goal_shoulder_pan - observations[2]) <= self.tolerance):
            
            rospy.logerr("YEAH!!! Robot arm reached goal position")

            self.reached_goal = True
            done = True

            rospy.sleep(3)

        # Return done when number of steps / iterations are completed
        if self.iteration == self.max_iterations - 1:
            done = True

        rospy.loginfo("FINISHED get _is_done")

        return done

    def _compute_reward(self, observations, done):
        rospy.logdebug("START _compute_reward")
        
        # Get the current location of the 3 joints
        cur_pos = np.array([self.joints[0], self.joints[1], self.joints[2]])
 
        # Calculate the step reward
        reward = 1/np.sqrt(np.sum(np.square(cur_pos - self.goal_pos)))

        # If goal was reached, add goal reward
        if self.reached_goal:
            reward += self.reached_goal_reward

        rospy.logdebug("END _compute_reward")

        # Publish the step reward
        self._publish_step_reward(reward)

        # Bump counters
        self.iteration += 1
        self.total_iteration += 1

        return reward

    def _publish_step_reward(self, reward):
        '''
        Publish the step reward after each step.
        '''
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = self.total_iteration
        reward_msg.episode_reward = reward
        self.pub_step_reward.publish(reward_msg)

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.steps_beyond_done = None
        self.iteration = 0
        self.reached_goal = False

    def _set_init_pose(self):
        """
        Sets joints to initial position [0,0,0]
        :return:
        """

        self.check_publishers_connection()

        # Reset Internal pos variable
        #
        # Use .copy() to prevent overwriting self.init_pos
        self.init_internal_vars(self.init_pos.copy())
        self.move_joints(self.pos)
        rospy.sleep(2)