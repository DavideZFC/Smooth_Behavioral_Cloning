from classes.classes import *
from functions.functions import *
import pickle
import numpy as np
import datetime
import os


# here the parameters of the experiment
env_name = 'BipedalWalker-v3'#'ContinuousCartPole'
Ns = np.linspace(200, 2000, 5)
seeds = 20 # number of random seeds to use
test_episodes = 40 # number of episodes to test the environment
FIRST_SEED = 400 # select the first seed of the sequence
full_noise = False # yes to apply global_noise on any action by default

# parameters about what to try
noise_levels = [0.05, 0.1, 0.2] # which standard deviations for the noise to try
regul_levels = [0.0005, 0.001, 0.002] # which levels of regularization to use for the imitator network

global_ep = 10 #number of epochs of training
global_layers = [50, 50] # number of neurons and layers in the imitator networks
global_noise = 0.0 #0.3 per walker
global_lam = 1.0 #discount factor
AR_parameter = 0.75 #parameter of the autoregressive model to make the noise time correlated

state_noise = False # yes to apply a noise also on the state

# here we store previous information
info = {}
info['test_len'] = test_episodes
info['env'] = env_name
info['n_seeds'] = seeds
info['test_episodes'] = test_episodes
info['AR_parameter'] = AR_parameter

info['global_ep'] = global_ep
info['global_layers'] = global_layers
info['global_noise'] = global_noise
info['NS'] = Ns
info['full_noise'] = full_noise
info['lam'] = global_lam


tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
dir = 'results/'+env_name+'_'+tail+'/'
os.mkdir(dir)


print('-----------------------------------')
print('Everything loaded, experiment starts')
print('-----------------------------------')


DO_NOISES = not True
DO_REG = not True

if full_noise:
    expert_policy, expert_ret = expert_policy_loader(env_name, noise=global_noise)
else:
    expert_policy, expert_ret = expert_policy_loader(env_name, noise=0.0, episodes=1)

info['expert_ret'] = expert_ret

with open(dir+"info", "wb") as fp:   #Pickling
   pickle.dump(info, fp)

print('-----------------------------------')
print('Expert loaded and evaluated: {}'.format(expert_ret))
print('-----------------------------------')

## Here i store a file to say what will be done in this experiment

content = ['BC']
if DO_NOISES:
    for i in range(len(noise_levels)):
        noise = noise_levels[i]
        content.append('noise_'+str(noise))

if DO_REG:
    for i in range(len(regul_levels)):
        regul = regul_levels[i]
        content.append('_regul_'+str(regul))

with open(dir+"content", "wb") as fp:   #Pickling
   pickle.dump(content, fp)





for seed in range(seeds):
    sub_dir = dir+'seed'+str(seed)+'/'
    os.mkdir(sub_dir)

    # Do BC

    name = 'BC'
    save_name = sub_dir+name
    get_performance(env_name, expert_policy, Ns, noise=global_noise, layers=global_layers, epochs=global_ep, seed=FIRST_SEED+seed, ep=test_episodes, full_noise=full_noise, save=True, save_name=save_name, lam=global_lam)
    print(name+' finished')

    # Do Noises

    if DO_NOISES:
        for i in range(len(noise_levels)):
            noise = noise_levels[i]
            name = 'noise_'+str(noise)
            save_name = sub_dir+name
            get_performance(env_name, expert_policy, Ns, seed=FIRST_SEED+seed, noise=noise, ep=test_episodes, full_noise=full_noise, save=True, save_name=save_name, state_noise=state_noise)
            print(name+' finished')
    
    # Do regularizations

    if DO_REG:
        for i in range(len(regul_levels)):
            reg = regul_levels[i]
            name = '_regul_'+str(reg)
            save_name = sub_dir+name
            get_performance(env_name, expert_policy, Ns, seed=seed, noise=global_noise, layers=global_layers, epochs=global_ep, regul=reg, ep=test_episodes, full_noise=full_noise, save=True, save_name=save_name, lam=global_lam, AR_parameter=AR_parameter)
            print(name+' finished')

    print('Seed {} is completed'.format(seed))


