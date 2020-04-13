import numpy as np
import gym
import time
import matplotlib.pyplot as plt

def eval_state_action(V, s, a, gamma=0.99):
    return np.sum([p * (rew + gamma*V[next_s]) for p, next_s, rew, _ in env.P[s][a]])

def value_iteration(V_estimates, eps=0.0001):
    V = np.zeros(nS)
    it = 0
    while True:
        V_estimates.append([])
        it += 1
        delta = 0
        for s in range(nS):
            old_v = V[s]
            V[s] = np.max([eval_state_action(V, s, a) for a in range(nA)])
            delta = max(delta, np.abs(old_v - V[s]))
            V_estimates[-1].append(V[s])
        if delta < eps:
            break
    print('Converged after %i iterations'%it)
    return V


def get_policy(V):
    policy = np.zeros(nS)
    for s in range(nS):
        policy[s] = np.argmax([eval_state_action(V, s, a) for a in range(nA)])
    return policy


def run_episodes(env, policy, num_games=100):
    tot_rew = 0
    state = env.reset()
    for _ in range(num_games):
        done = False
        while not done:
            next_state, reward, done, _ = env.step(policy[state])                
            state = next_state
            tot_rew += reward 
            if done:
                state = env.reset()
    print('Average return per episode: %.2f'%(tot_rew/num_games))


def render_episode(env, policy):
    tot_rew = 0
    state = env.reset()
    env.render()
    time.sleep(1)
    done = False
    while not done:
        next_state, reward, done, _ = env.step(int(policy[state]))
        # cast action to int necessary for env.render()
        state = next_state
        tot_rew += reward
        env.render()
        time.sleep(1)
    print('Reward: %i'%tot_rew)

            
if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env = env.unwrapped
    nA = env.action_space.n
    nS = env.observation_space.n
    V_estimates  = [[0] * nS]
    V = value_iteration(V_estimates)
    policy = get_policy(V)

    run_episodes(env, policy)

    # visualise some games
    # for ep in range(5):
    #     print('\n\n\nEpisode: %i'%ep)
    #     render_episode(env, policy)

    # display V-function and policy for FrozenLake
    print(V.reshape((4,4)))
    print(policy.reshape((4,4)))

    # plot V-function iteration estimates
    plt.plot(V_estimates)
    plt.show()

    env.close()


