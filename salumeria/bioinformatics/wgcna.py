# a humble implementation of WCGNA
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx

'''
based on Zhang and Hovarth 2005
'''

def weighted_coexpression_similarity(expression_values, signed=True, method="pearson", threshold = 'default'):

    cor_mat = expression_values.transpose().corr(method)
    if signed:
        if threshold == 'default':
            threshold = 12
        return ((cor_mat + 1)/2)**threshold
    if threshold == 'default':
        threshold = 6
    return abs(cor_mat)**threshold

def TOM(a):
    k = np.sum(a) # connectivity of nodes
    l = np.dot(a, a) # dot of the similarity matrix
    K = np.min(np.stack((np.tile(k, (len(k),1)), np.tile(k, (len(k),1)).T), axis = 2), axis = 2) # kmin{i,j}
    tom = (l+a)/(K + 1 + a)
    np.fill_diagonal(TOM.values, 1)
    return tom

def dissimilarity(tom):
    return (1-tom)

def clusterize(dis, method='average', optimal_ordering=True ,**kwargs):
    dendro = dendrogram(linkage(dis, method=method, optimal_ordering=optimal_ordering), **kwargs) # use kwargs for dendro args
    cluster_map = dict(zip(list(dis.index[dendro['leaves']]), dendro['leaves_color_list']))
    return(cluster_map)

def make_network(tom, threshold=0):
    # diagonal of TOM should be 0
    np.fill_diagonal(tom.values, 0)
    nt = nx.from_pandas_adjacency(tom)
    nt.remove_edeges_from([(k,j) for k,j,i in nt.edges(data='weight') if i < threshold])
    return nt



## docstrings

weighted_coexpression_similarity.__doc__ = '''
calculate a weighted similarity matrix to be used for gene expression values.
performs only soft thresholding (hard to be implemented).
---
parameters
    expression_values:  mandatory
    signed:             boolean, wether the coexpression should be signed or unsigned
'''
