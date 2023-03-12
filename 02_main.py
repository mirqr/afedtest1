from cgitb import reset
import sys
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from joblib import dump, load

from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.utils.utility import precision_n_scores, standardizer
from sklearn.metrics import auc, f1_score, roc_auc_score, roc_curve

from autocustom import AutoEncoderCustom

from datautil import *


def get_dataset(dataset_name: str):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test,                             y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset_name == 'kminst':
        x_train = np.load('./kmnist/kmnist-train-imgs.npz')['arr_0']
        x_test = np.load('./kmnist/kmnist-test-imgs.npz')['arr_0']
        y_train = np.load('./kmnist/kmnist-train-labels.npz')['arr_0']
        y_test = np.load('./kmnist/kmnist-test-labels.npz')['arr_0']
    else:
        raise('Error')

    return x_train, y_train, x_test, y_test


def get_model(devs, start, end, reset=False):
    for dev in devs[start:end]:
        dev.model_fit(
            #OCSVM(contamination=outlier_fraction,cache_size=1000),
            #AutoEncoder(contamination=outlier_fraction, hidden_neurons = [16, 8, 16], dropout_rate=0.1, verbose=1, preprocessing=False),
            AutoEncoderCustom(contamination=outlier_fraction, hidden_neurons = [16, 8, 16], dropout_rate=0.1, verbose=0, preprocessing=False),
            num_nodes_per_class=num_nodes_per_class,reset=reset)

def broad(devs, start, end, reset=False):
    get_model(devs, 0, None)

    for i in devs[start:end]:
        i.set_other_models(devs, reset=reset)
        #for i in devs:
        #i.write_other_models(devs)


if __name__ == "__main__":
    args = sys.argv[1:]
    start = int(args[0])
    end = int(args[1])
    print('start',start,'end',end)
    # config
    dataset_name = "mnist"
    dataset_name = "fashion_mnist"
    dataset_name = "kminst"
    outlier_fraction = 0.1


    x_train, y_train, x_test, y_test = get_dataset(dataset_name)

    n_features = np.prod(x_train.shape[1:])
    x_train = x_train.reshape(x_train.shape[0], n_features) / 255.0
    x_test = x_test.reshape(x_test.shape[0], n_features) / 255.0

    num_classes = np.unique(y_train).size
    num_nodes_per_class = 9  # p parameter, set (to num_class - 1) * d

    print(x_train.shape)
    print(x_test.shape)

    # dict of dataset, key is client name 
    datasets = partition(x_train,y_train,num_nodes_per_class, outlier_fraction)

    devs = []
    for name, data, in datasets.items():
        #print(name,dataset_name)
        x = data.iloc[:, :28*28]
        y_class = data['y_class']
        devs = devs + [Dev(name, dataset_name, x, y_class)]

    

    get_model(devs, start, end,True)
    #broad(devs, start, end,True)
    