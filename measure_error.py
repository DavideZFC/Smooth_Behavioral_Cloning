import gym
import numpy as np
import pickle
import torch
from functions.functions import *


def bootstrap_ci(x, conf=0.95, resamples=10000):
    means = [np.mean(x[np.random.choice(x.shape[0], size=x.shape[0], replace=True), :], axis=0) for _ in range(resamples)]
    low = np.percentile(means, (1-conf)/2 * 100, axis=0)
    high = np.percentile(means, (1 - (1-conf)/2) * 100, axis=0)
    low = np.nan_to_num(low)
    high = np.nan_to_num(high)
    return low, high


dir = 'results\Pendulum-v0_22_07_11-12_43_'+'/'
# elect which experiment you want to plot

with open(dir+"content", "rb") as fp:
   content = pickle.load(fp)
with open(dir+"info", "rb") as fp:
   info = pickle.load(fp)

test_episodes = info['test_len']
env_name = info['env']
seeds = info['n_seeds']
test_episodes = info['test_episodes']

global_ep = info['global_ep']
global_layers = info['global_layers']
global_noise = info['global_noise']
Ns = info['NS']
full_noise = info['full_noise']

if full_noise:
    expert_policy, expert_ret = expert_policy_loader(env_name, noise=global_noise)
else:
    expert_policy, expert_ret = expert_policy_loader(env_name, noise=0.0)

'''
load stuff
'''

env = gym.make(env_name)
N = 10000

print('-----------------------------------')
print('Everything loaded, start collecting datasets')
print('-----------------------------------')

lam = 0.9

states, actions = collect_data(env, expert_policy, N, noise = 0.0, lam=0.9)

torch_grid = torch.from_numpy(states).float()
y, _ = expert_policy.predict(states)


for c in content:
    mean_matrix = np.zeros((seeds, len(Ns)))
    for seed in range(seeds):
        error_vector = np.zeros(len(Ns))
        subdir = dir+'seed'+str(seed)+'/'

        for i in range(len(Ns)):
            name = subdir+c+'_step_{}.pt'.format(i)
            net = torch.load(name)

            y_pred = net.forward(torch_grid).detach().numpy()

            mse = np.mean((y-y_pred)**2)**0.5
            
            mean_matrix[seed, i] = mse
    low, high = bootstrap_ci(mean_matrix)
    print(c+' lower bound')
    print(low)
    print(c+' upper bound')
    print(high)
    #plt.plot(Ns, error_vector, label=c)

   
