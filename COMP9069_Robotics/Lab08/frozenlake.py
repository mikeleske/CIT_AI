import gym

# create the environment
env = gym.make("FrozenLake-v0")
print(env.action_space)
print(env.observation_space)

# reset (initialise) the environment before starting
env.reset()
env.render()

done = False
while not done:
   action = int(input("Enter input: "))
   new_obs, rew, done, info = env.step(action)
   env.render()
   print(new_obs)

# close the environment
env.close()