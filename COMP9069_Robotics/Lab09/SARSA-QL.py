'''

Code adapted from:
https://github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python/blob/master/Chapter04/SARSA%20Q_learning%20Taxi-v2.py

'''

import numpy as np 
import gym
import matplotlib.pyplot as plt


def eps_greedy(Q, s, eps=0.1):
    '''
    Epsilon greedy policy
    '''
    if np.random.uniform(0,1) < eps:
        # Choose a random action
        return np.random.randint(Q.shape[1])
    else:
        # Choose the action of a greedy policy
        return greedy(Q, s)


def greedy(Q, s):
    '''
    Greedy policy

    return the index corresponding to the maximum action-state value
    '''
    return np.argmax(Q[s])


def run_episodes(env, Q, num_episodes=100):
    '''
    Run some episodes to test the policy
    '''
    tot_rew = []
    state = env.reset()

    for _ in range(num_episodes):
        done = False
        game_rew = 0

        while not done:
            # select a greedy action
            next_state, rew, done, _ = env.step(greedy(Q, state))

            state = next_state
            game_rew += rew 
            if done:
                state = env.reset()
                tot_rew.append(game_rew)

    return np.mean(tot_rew)


def Q_learning(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    nA = env.action_space.n
    nS = env.observation_space.n

    # Initialize the Q matrix
    # Q: matrix nS*nA where each row represents a state and each column represents a different action
    Q = np.zeros((nS, nA))
    test_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        
        # decay the epsilon value until it reaches the threshold of 0.01
        if eps > 0.01:
            eps -= eps_decay

        # loop the main body until the environment stops
        while not done:
            # select an action following the eps-greedy policy
            action = eps_greedy(Q, state, eps)

            next_state, rew, done, _ = env.step(action) # Take one step in the environment

            # Q-learning update the state-action value (get the max Q value for the next state)
            Q[state][action] = Q[state][action] + lr*(rew + gamma*np.max(Q[next_state]) - Q[state][action])

            state = next_state

        # Test the policy every 200 episodes and print the results
        if (ep % 200) == 0:
            test_rew = run_episodes(env, Q, 1000)
            print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)
            
    return Q, test_rewards


def SARSA(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    nA = env.action_space.n
    nS = env.observation_space.n

    # Initialize the Q matrix
    # Q: matrix nS*nA where each row represents a state and each column represents a different action
    Q = np.zeros((nS, nA))
    test_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False

        # decay the epsilon value until it reaches the threshold of 0.01
        if eps > 0.01:
            eps -= eps_decay

        action = eps_greedy(Q, state, eps) 

        # loop the main body until the environment stops
        while not done:
            next_state, rew, done, _ = env.step(action) # Take one step in the environment

            # choose the next action (needed for the SARSA update)
            next_action = eps_greedy(Q, next_state, eps) 
            # SARSA update
            Q[state][action] = Q[state][action] + lr*(rew + gamma*Q[next_state][next_action] - Q[state][action])

            state = next_state
            action = next_action

        # Test the policy every 200 episodes and print the results
        if (ep % 200) == 0:
            test_rew = run_episodes(env, Q, 1000)
            print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)

    return Q, test_rewards


def dp(env, gamma=0.95):

    nS = env.observation_space.n
    nA = env.action_space.n

    def eval_state_action(V, s, a, gamma=0.99):
        return np.sum([p * (rew + gamma*V[next_s]) for p, next_s, rew, _ in env.P[s][a]])

    def value_iteration(eps=0.0001):
        V = np.zeros(nS)
        while True:
            delta = 0
            for s in range(nS):
                old_v = V[s]
                V[s] = np.max([eval_state_action(V, s, a) for a in range(nA)])
                delta = max(delta, np.abs(old_v - V[s]))
            if delta < eps:
                break
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
        print('Average return per episode: %.4f'%(tot_rew/num_games))

    V = value_iteration()
    policy = get_policy(V)
    run_episodes(env, policy, 1000)


if __name__ == '__main__':
    env = gym.make('Taxi-v3')

    print("DP (Value Iteration):")
    dp(env)

    print("\n\nQ-Learning:") 
    Q_qlearning, test_rew_qlearning = Q_learning(env, lr=0.1, num_episodes=5000, eps=0.8, gamma=0.95, eps_decay=0.001)

    print("\n\nSARSA:") 
    Q_sarsa, test_rew_sarsa = SARSA(env, lr=0.1, num_episodes=5000, eps=0.8, gamma=0.95, eps_decay=0.001)

    plt.plot(test_rew_qlearning, label='Q-Learning')
    plt.plot(test_rew_sarsa, color='r', label='SARSA')
    plt.legend()
    plt.show()

    env.close()


