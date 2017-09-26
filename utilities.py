import numpy as np
import re
import tensorflow as tf
import math
from matplotlib import pyplot

def create_minibatch(data, batch_size=20):
    rng = np.random.RandomState()
    while True:
        ixs = rng.choice(data.shape[0], size=batch_size, replace=False)
        yield data[ixs], ixs

def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def variable_parser(var_list, prefix):
    ret_list = []
    for var in var_list:
        varname = var.name
        varprefix = re.split(r'[:/_]+', varname)[0]
        if varprefix == prefix:
            ret_list.append(var)
    return ret_list

def weights_initialisation(inputSize, outputSize, he_flag=False, relu_flag=False):
    init_range = np.sqrt(6.0 / (inputSize + outputSize*(1-np.array([he_flag]).astype(int))))
    if relu_flag:
        init_range *= np.sqrt(2)
    return tf.random_uniform([inputSize, outputSize], minval=-init_range, maxval=init_range, dtype=tf.float32)

def biases_initialisation(outputSize):
    return tf.random_normal([outputSize], stddev=1)*0

def sample_gumbel(shape, epsilon=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + epsilon) + epsilon)

def gumbel_softmax_sample(logits, sample_uniform_gumbel, annealed_tau):
    y = logits + sample_uniform_gumbel
    return tf.nn.softmax( y / annealed_tau)

def print_top_words(beta, feature_names, n_top_words=20, title=''):
    print('---------------Printing the Topics------------------')
    if title != '':
        f = open("C:/UCL/Thesis/IWAE_1024/" + title, "w")
    for i in range(len(beta)):
        if title != '':
            if i < len(beta)-1:
                f.write(" ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]])+"\n")
            else:
                f.write(" ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
        print(i, np.sum(np.sort(beta[i,:])[:-n_top_words-1:-1]), " ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words-1:-1]]))
    print('---------------End of Topics------------------')
    if title != '':
        f.close()

def activity(h_mu_list_array, gaussian_list, title=''):
    mu_var = np.var(h_mu_list_array, 0)
    mu_var_order = np.argsort(mu_var)[::-1]
    g_kl = np.mean(np.array(gaussian_list), (0, 1))
    pyplot.figure()
    pyplot.semilogy(range(len(mu_var_order)), mu_var[mu_var_order], linestyle='None', marker='o', color='r')
    pyplot.semilogy(range(len(mu_var_order)), g_kl[mu_var_order], linestyle='None', marker='d', color='b')
    pyplot.xlabel('Unit')
    pyplot.ylabel('Activity')
    pyplot.legend(('Activity', 'KL-divergence'), loc='upper right', framealpha=0.0)
    if title != '':
        pyplot.savefig('C:/UCL/Thesis/IWAE_1024/' + str(title) + '.png')
