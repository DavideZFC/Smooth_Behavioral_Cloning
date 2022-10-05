import pickle

import numpy as np
import matplotlib.pyplot as plt


def bootstrap_ci(x, conf=0.95, resamples=10000):
    means = [np.mean(x[np.random.choice(x.shape[0], size=x.shape[0], replace=True), :], axis=0) for _ in range(resamples)]
    low = np.percentile(means, (1-conf)/2 * 100, axis=0)
    high = np.percentile(means, (1 - (1-conf)/2) * 100, axis=0)
    low = np.nan_to_num(low)
    high = np.nan_to_num(high)
    return low, high

def normal_ci(x):
   mu = np.mean(x, axis=0)
   sigma = np.std(x, axis=0)
   n = x.shape[0]
   return mu - sigma/(n**0.5), mu + sigma/(n**0.5)



dir = 'results\ContinuousCartPole_22_07_27-15_20_'+'/'
# elect which experiment you want to plot

with open(dir+"content", "rb") as fp:
   content = pickle.load(fp)

with open(dir+"info", "rb") as fp:
   info = pickle.load(fp)


Ns = info['NS']
seeds = info['n_seeds']


for c in content:
   print(c)
   mean_matrix = np.zeros((seeds, len(Ns)))
   for seed in range(seeds):
      subdir = dir+'seed'+str(seed)+'/'
      means = np.load(subdir+c+'_means.npy')
      mean_matrix[seed, :] = means
   #low, high = bootstrap_ci(mean_matrix)
   low, high = normal_ci(mean_matrix)

   if c == 'BC':
      plt.plot(Ns, (low+high)/2, label=c)
   else:
      plt.plot(Ns, (low+high)/2, label=c[1:])
   plt.fill_between(Ns, y1=low, y2=high, alpha=0.3)

plt.legend()
plt.title('Experiment on '+ info['env'])
plt.show()
        