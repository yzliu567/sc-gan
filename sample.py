import numpy as np
from tqdm import tqdm
import random
from sklearn.neighbors import NearestNeighbors
import argparse
def getnorm(val, min_val, max_val):
    ss = (val - min_val)/ (max_val - min_val)
    return np.clip(ss, 0, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dists',nargs='+', help="distance files to sample")
    parser.add_argument('--exp_setting', type=str, help="output file name")
    args = parser.parse_args()
    nums = len(args.dists)
    scores = []
    minmax = []
    basedir = './distances/'
    for i in range(nums):
        score_i = np.loadtxt(basedir + args.dists[i]).squeeze()
        s_min = np.percentile(score_i, 0.5)
        s_max = np.percentile(score_i, 99.5)
        score_i = getnorm(score_i, s_min, s_max)
        minmax += [s_min, s_max]
        scores.append(score_i)
    # save min and max
    minmax = np.array(minmax)
    np.savetxt(basedir + args.exp_setting + '_min_max.txt', minmax)

    # multi-dimensional uniform sample
    X = np.stack(scores, axis=1)
    all_num = 250000
    query = np.random.uniform(0,1,(all_num, nums))
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(query)
    # randomly choose from nearest 10 samples
    choice = np.random.randint(0,10,(all_num))
    idx = np.arange(all_num)
    indices = indices[idx,choice]
    np.savetxt(basedir + 'index_' + args.exp_setting + '.txt', indices)