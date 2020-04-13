import gym

# create the environment
env = gym.make("Taxi-v3")
print(env.action_space)
print(env.observation_space)

# reset (initialise) the environment before starting
env.reset()
env.render()

done = False
tot_rew = 0
while not done:
   action = int(input("Enter input: "))
   new_obs, rew, done, info = env.step(action)
   tot_rew += rew
   env.render()
   print(tot_rew)

# close the environment
env.close()