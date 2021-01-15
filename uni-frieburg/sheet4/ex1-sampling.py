import random
import time
import scipy.stats
from math import *
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def sample_normal_distribution(mu,vari):
    try:
        sigma = np.sqrt(vari)
        x = 0.5*np.sum(np.random.uniform(-sigma, sigma, 12))
        return mu+x
    except Exception as e:
        print(e)

def sample_normal_rejection(mu,vari):
    """
    for this, we need F(x). We get "F" using `scipy.stats.norm(mu,sigma)`
    and "F(x)" by taking its pdf at x; i.e., `scipy.stats.norm(mu,sigma).pdf(x)`
    """
    try:
        while True:
            sigma = np.sqrt(vari)
            interval = 5*sigma
            max_density = scipy.stats.norm(mu,sigma).pdf(mu)
            x = random.uniform(mu-interval, mu+interval) #IDK why we didn't go for (-sigma, sigma)
            y = random.uniform(0,max_density)
            if(y<=scipy.stats.norm(mu,sigma).pdf(x)):
                break
        return x
    except Exception as e:
        print(e)

def sample_normal_boxmuller(mu,vari):
    try:
        sigma = np.sqrt(vari)
        u1, u2 = random.uniform(0.0,1.0), random.uniform(0.0,1.0)
        x = cos(2*pi*u1) * np.sqrt(-2*log(u2))
        return x + sigma + mu
    except Exception as e:
        print(e)

def evaluate_sampling_time(mu,vari,n_samples,func):
    tic = time.time()
    for _ in range(n_samples):
        func(mu,vari)
    toc = (time.time() - tic)/n_samples
    print(f'{func.__name__}\t {toc}')

def evaluate_sampling_dist(mu,vari,n_samples,func):
    n_bins = 100
    """
    A histogram displays numerical data by grouping data into "bins" of equal width.
    Each bin is plotted as a bar whose height corresponds to how many data points are
    in that bin. Bins are also sometimes called "intervals", "classes", or "buckets"
    """
    sigma = np.sqrt(vari)
    samples = []
    for i in range(n_samples):
        samples.append(func(mu,vari))
    print(f'{func.__name__}: \t u= {np.mean(samples)}\t std= {np.std(samples)}')
    plt.figure()
    count, bin, _ = plt.hist(samples, n_bins)
    plt.plot(bin, scipy.stats.norm(mu,sigma).pdf(bin), linewidth=2, color='r') #IDK what this gives
    plt.xlim([mu - 5*sigma, mu + 5*sigma])
    plt.title(func.__name__)

def main():
    mu, vari = 0,1
    sample_functions = [sample_normal_distribution,
                        sample_normal_rejection,
                        sample_normal_boxmuller,
                        np.random.normal]
    for func in sample_functions:
        evaluate_sampling_time(mu,vari,1000,func)
        evaluate_sampling_dist(mu,vari,1000,func)
    plt.show()

if __name__ == "__main__":
    main()
