import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
import scipy.io as io



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def construct_graph(dataset, features, topk):
    fname = './raw_data/drug_data/' + dataset + '/dis_knn/tmp.txt'
    print(fname)
    f = open(fname, 'w')
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()


def generate_knn(dataset):

        topk = 10
        # data = io.loadmat('./raw_data/drug_data/' + dataset + '/' +'Cdataset.mat')
        # construct_graph(dataset, data['disease'], topk)
        # data = np.loadtxt('./raw_data/drug_data/' + dataset + '/' +'lagcn/dis_sim.csv', delimiter=",")
        # construct_graph(dataset, data, topk)

        # data = np.loadtxt('./raw_data/drug_data/' + dataset + '/drug_sim.txt', dtype=str, delimiter="\t")
        # data = data[1:, 1:]
        # data = data.astype('float64')
        data = pd.read_csv('./raw_data/drug_data/' + dataset + '/dis_sim.txt', index_col=0, delimiter='\t')
        data = data.values

        construct_graph(dataset, data, topk)
        f1 = open('./raw_data/drug_data/' + dataset + '/dis_knn/tmp.txt', 'r')
        f2 = open('./raw_data/drug_data/' + dataset + '/dis_knn/c'+str(topk)+'.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            print(line)
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{}\t{}\n'.format(start, end))
        f2.close()

'''generate KNN graph'''

generate_knn('lrssl')