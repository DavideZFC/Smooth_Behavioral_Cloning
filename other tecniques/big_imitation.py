import numpy as np
import matplotlib.pyplot as plt
import gym
from sklearn.neighbors import KNeighborsRegressor
from functions.functions import *
from sklearn.cluster import KMeans

def get_policy(states, actions):
    policy = KNeighborsRegressor(n_neighbors=1)
    policy.fit(states, actions)
    return policy

def test(env, policy, N, render=False):
    ret = 0
    for i in range(N):
        if i==N-1:
            render = True
        done = False
        state = env.reset()
        while not done:
            try:
                state, reward, done, _ = env.step(policy.predict(state.reshape(1,-1))[0])
            except:
                print(state.reshape(1,-1))
                print(policy.predict(state.reshape(1,-1)))
                state, reward, done, _ = env.step(policy.predict(state.reshape(1,-1)))
            ret += reward
            if render:
                env.render()
        env.close()
    return ret/N

def collect_data(env, N, policy='random', lam=1.0, noise=0.2):
    states = np.zeros((N, env.observation_space.shape[0]))
    n_steps = 0
    while n_steps<N:
        done = False
        state = env.reset()
        while not done and n_steps<N:
            if policy=='random':
                state, reward, done, _ = env.step(env.action_space.sample())
            else:
                try:
                    action = policy.predict(state.reshape(1,-1))
                    action = (1-noise)*action + noise*env.action_space.sample()
                    state, reward, done, _ = env.step(action)
                except:
                    action = policy.predict(state.reshape(1,-1))[0]
                    action = (1-noise)*action + noise*env.action_space.sample()
                    state, reward, done, _ = env.step(action)
            states[n_steps,:] = state
            n_steps += 1
            if np.random.uniform()>lam:
                done = True
    return states




env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)
n_clusters = 20


states = collect_data(env,5000, lam=0.9)
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(states)

centers = kmeans.cluster_centers_

expert_policy, expert_ret = expert_policy_loader(env_name, noise=0.0)

try:
    best_actions, _ = expert_policy.predict(centers)
except:
    best_actions = expert_policy.predict(centers)
policy = get_policy(centers, best_actions)

print('The return is '+str(test(env,policy,N=5)))

iter = 3
for _ in range(iter):
    states = collect_data(env, 10000, lam=1, policy=policy)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(states)

    new_centers = kmeans.cluster_centers_
    centers = np.append(centers, new_centers, axis=0)

    try:
        best_actions, _ = expert_policy.predict(centers)
    except:
        best_actions = expert_policy.predict(centers)
    policy = get_policy(centers, best_actions)


    print('The return is '+str(test(env,policy,N=5)))
