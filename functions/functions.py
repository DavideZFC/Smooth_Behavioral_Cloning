import numpy as np
import matplotlib.pyplot as plt
import gym
from stable_baselines3 import SAC, DDPG
from classes.classes import *


def expert_policy_loader(env_name, episodes=20, noise = 0.0):
    '''
    Loads and tests the expert policy
    '''

    if env_name == 'ContinuousCartPole':
        env = ContinuousCartPoleEnv()
    else:
        env = gym.make(env_name)

    if env_name == 'LunarLanderContinuous-v2':
        expert_policy = SAC.load('trained_agents/sac_lunar_stable_baselines_3')
        action_size = env.action_space.shape[0]
        sb_expert = True
    
    elif env_name == 'Pendulum-v0':
        expert_policy = DDPG.load('trained_agents/ddpg_pendulum')
        action_size = env.action_space.shape[0]
        sb_expert = True

    elif env_name == 'BipedalWalker-v3':
        action_size = env.action_space.shape[0]
        state_size = env.observation_space.shape
        expert_policy = Actor_Model(state_size, action_size)
        Actor_name = 'trained_agents/BipedalWalker-v3_PPO_Actor.h5'
        expert_policy.Actor.load_weights(Actor_name)
        expert_policy = pytorchActor(expert_policy)
        sb_expert = False

    elif env_name == 'MountainCarContinuous-v0':
        action_size = env.action_space.shape[0]
        state_size = env.observation_space.shape
        expert_policy = irregular_policy()
        sb_expert = False

    elif env_name == 'ContinuousCartPole':
        action_size = env.action_space.shape[0]
        state_size = env.observation_space.shape
        expert_policy = DDPG.load("trained_agents/ddpg_cart_pole")
        sb_expert = True

    else:
        print("Sorry, environment not supported")

    try:
        N = len(noise)
        multi_noise = True
    except:
        multi_noise = False

    if multi_noise:
        final_resu = np.zeros(N)
        for i in range(N):
            returns = []
            for ep in range(episodes):
                this_return = 0
                done = False
                state = env.reset()
                while not done:

                    e = np.random.normal(0,noise[i],size=action_size)
                    if sb_expert:
                        action = expert_policy.predict(state)[0]    
                    else:
                        action = expert_policy.predict(state)
                    
                    if (not env_name=='LunarLanderContinuous-v2') or state[1]>0:
                        new_state, reward, done, _ = env.step(action + e)
                    else:
                        new_state, reward, done, _ = env.step(action)
                    state = new_state
                    this_return += reward
                returns.append(this_return)
            ret = np.mean(returns)
            final_resu[i] = ret
        return expert_policy, final_resu
    else:
        returns = []
        for ep in range(episodes):
            this_return = 0
            done = False
            state = env.reset()
            while not done:

                e = np.random.normal(0,noise,size=action_size)
                if sb_expert:
                    action = expert_policy.predict(state)[0]    
                else:
                    action = expert_policy.predict(state)
                
                if (not env_name=='LunarLanderContinuous-v2') or state[1]>0:
                    new_state, reward, done, _ = env.step(action + e)
                else:
                    new_state, reward, done, _ = env.step(action)
                state = new_state
                this_return += reward
            returns.append(this_return)

        return expert_policy, np.mean(returns)











def evaluate_policy(env, policy, ep = 20, render=False):
    returns = []
    for i in range(ep):
        ret = 0
        done = False
        state = env.reset()
        while (not done):
            action = policy.prediction(state[np.newaxis,:])[0]
            # action = policy.prediction(state)[0]
            new_state, reward, done, _ = env.step(action)
            state = new_state
            ret += reward
            if(render and i==0):
                env.render()
        returns.append(ret)

    if (render):
        env.close()
    return np.mean(returns), np.std(returns)/ep






def collect_data(env, policy, N, noise=0, sb_expert=True, env_name='BipedalWalker-v3', full_noise=True, lam=1, eta=0, state_noise=False):
    N = int(N)
    states = np.zeros((N, env.observation_space.shape[0]))
    action_dim = env.action_space.shape[0]
    actions = np.zeros((N, action_dim))
    n_steps = 0

    noise_list = [noise*np.random.randn(action_dim)]

    if env_name == 'LunarLanderContinuous-v2':
        while n_steps<N:
            done = False
            state = env.reset()
            while (not done) and (n_steps<N):                
                if sb_expert:
                    action = policy.predict(state)[0]
                else:
                    action = policy.predict(state)

                if full_noise and state[1]>0: # we take the noise both
                    action += noise*np.random.randn(action_dim)
                    new_state, _, done, _ = env.step(action)
                elif state[1]>0:
                    print('executed -> full noise = False')
                    new_state, _, done, _ = env.step(action + noise*np.random.randn(action_dim))
                else:
                    new_state, _, done, _ = env.step(action)

                states[n_steps,:] = state
                actions[n_steps,:] = action
                state = new_state
                n_steps += 1
    
    else:
        while n_steps<N:
            done = False
            state = env.reset()
            while (not done) and (n_steps<N):
                if state_noise:
                    if sb_expert:
                        action = policy.predict(state + np.random.normal(0,noise,size=env.observation_space.shape[0]))[0]
                    else:
                        action = policy.predict(state + np.random.normal(0,noise,size=env.observation_space.shape[0]))                   
                else:
                    if sb_expert:
                        action = policy.predict(state)[0]
                    else:
                        action = policy.predict(state)

                if state_noise:
                    new_state, _, done, _ = env.step(action)
                else:
                    if full_noise: # in this case, we take the noise also in the dataset
                        e = (1-eta)*noise*np.random.normal(0,noise,size=action_dim) + eta*noise_list[-1]
                        action += e
                        new_state, _, done, _ = env.step(action)
                    else:
                        new_state, _, done, _ = env.step(action + noise*np.random.normal(0,noise,size=action_dim))
                try:
                    states[n_steps,:] = state[0,:]
                except:
                    states[n_steps,:] = state
                
                try:
                    actions[n_steps,:] = action[:,0]
                except:
                    actions[n_steps,:] = action
                    
                state = new_state
                n_steps += 1
                if np.random.binomial(1,p=lam) < 1:
                    done = True   

    return states, actions







def get_performance(env_name, expert_policy, Ns, seed, noise=0, epochs=10, layers=[50,50], regul=0, ep=20, lam=1, full_noise=True, save=True, save_name=0, AR_parameter=0, state_noise=False):
    
    if env_name == 'ContinuousCartPole':
        env = ContinuousCartPoleEnv()
    else:
        env = gym.make(env_name)
    sb_expert = (env_name == 'LunarLanderContinuous-v2' or env_name == 'Pendulum-v0' or env_name == 'ContinuousCartPole')

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    iter = len(Ns)

    states_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    means = []
    stds = []
    for i in range(iter):
        N = int(Ns[i])
        if i==0:
            states, actions = collect_data(env, expert_policy, N, noise = noise, sb_expert=sb_expert, env_name=env_name, full_noise=full_noise, lam=lam, eta=AR_parameter, state_noise=state_noise)
        else:
            new_states, new_actions = collect_data(env, expert_policy, N-Ns[i-1], noise = noise, sb_expert=sb_expert, env_name=env_name, full_noise=full_noise, lam=lam, eta=AR_parameter)
            states = np.append(states, new_states, axis=0)
            actions = np.append(actions, new_actions, axis=0)

        # defining and compiling the model
        model = Regression(inp=states_dim, out=action_dim, layers=layers)
        model.train(states, actions, weight_decay=regul, epochs=epochs)

        mean, std = evaluate_policy(env, model, ep=ep)
        means.append(mean)
        stds.append(std)

        if(save):
            model.save(save_name+'_step_'+str(i))

    means = np.array(means)
    stds = np.array(stds)
    np.save(save_name+'_means', means)



