import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, metrics
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2




class Autoencoder4(Model):
    def __init__(self, n_features, dropout_rate=0.1, hidden_neurons=[64, 32, 64]):
        super(Autoencoder4, self).__init__()
        #self.latent_dim = latent_dim

        self.n_features = n_features
        self.hidden_neurons = hidden_neurons
        self.dropout_rate = dropout_rate
        # cambia il default ([64,32,64]) cosi eviti di dover passare hidden nei tre punti client server
        self.hidden_neurons = hidden_neurons

        model = Sequential([
            # Input(shape=(28*28)),
            # Reshape((28, 28,1)),
            # Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            # Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),

            # Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            # Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            # Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'),
            # Flatten(),

        
            Input(shape=(28*28)),
            Reshape((28, 28,1)),
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),

            # at this point the representation is (4, 4, 8) i.e. 128-dimensional

            Conv2D(8, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(16, (3, 3), activation='relu'),
            UpSampling2D((2, 2)),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
            Flatten()
    
        ])

        self.mod = model

    def call(self, x):
        mod = self.mod(x)
        return mod



class Autoencoder3(Model): # USAVO QUESTO PRIMA DI 4
    def __init__(self, n_features, dropout_rate=0.1, hidden_neurons=[64, 32, 64]):
        super(Autoencoder3, self).__init__()
        #self.latent_dim = latent_dim

        self.n_features = n_features
        self.hidden_neurons = hidden_neurons
        self.dropout_rate = dropout_rate
        # cambia il default ([64,32,64]) cosi eviti di dover passare hidden nei tre punti client server
        self.hidden_neurons = hidden_neurons

        first = [
            # tf.keras.Input(shape=(n_features,)),
            #layers.Dropout(self.dropout_rate)
        ]
        internal = []

        for neu in self.hidden_neurons:
            internal.append(layers.Dense(neu, activation='relu'))
            #internal.append(layers.Dropout(self.dropout_rate))

        output = [
            layers.Dense(n_features, activation='sigmoid')
            #layers.Reshape((28, 28))
        ]
        self.mod = tf.keras.Sequential(first+internal+output)

    def call(self, x):
        mod = self.mod(x)
        return mod


class Autoencoder4(Model):
    def __init__(self, n_features):
        super(Autoencoder4, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Reshape((28, 28, 1)),
            #layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu',
                          padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
            ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(
                8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(
                16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'),
            layers.Flatten()
            ])
        
            

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder5(Model):
    def __init__(self, n_features):
        super(Autoencoder5, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Reshape((28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same')
        ])

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        self.decoder = tf.keras.Sequential([
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
            layers.Flatten()
        ])         

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder2(Model):
    def __init__(self, n_features, hidden_neurons=None,
                 hidden_activation='relu', output_activation='sigmoid',
                 loss=mean_squared_error, optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=False,
                 verbose=1, random_state=None):
        super(Autoencoder2, self).__init__()
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state

        #self.n_samples = n_samples
        self.n_features_ = n_features

        # default values
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32, 32, 64]

        # Verify the network design is valid
        if not self.hidden_neurons == self.hidden_neurons[::-1]:
            print(self.hidden_neurons)
            raise ValueError("Hidden units should be symmetric")
        self.hidden_neurons_ = self.hidden_neurons

        #check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',include_left=True)

        if np.min(self.hidden_neurons) > self.n_features_:
            raise ValueError("The number of neurons should not exceed "
                             "the number of features")
        # self.hidden_neurons_.insert(0, self.n_features_) #???? TODO senza questo e' uguale a autoencoder esempio tensorflow # vedi POP

        print(self.hidden_neurons_)

        # def _build_model(self):

        model = Sequential()
        # Input layer
        model.add(Dense(
            self.hidden_neurons_[0], activation=self.hidden_activation,
            input_shape=(self.n_features_,),
            # activity_regularizer=l2(self.l2_regularizer)  # funziona solo se attivo quello centrale
        ))
        model.add(Dropout(self.dropout_rate))

        # Additional layers
        for i, hidden_neurons in enumerate(self.hidden_neurons_, 1):
            model.add(Dense(
                hidden_neurons,
                activation=self.hidden_activation,
                # activity_regularizer=l2(self.l2_regularizer)
            )
            )
            model.add(Dropout(self.dropout_rate))

        # Output layers
        model.add(Dense(
            self.n_features_,
            activation=self.output_activation,
            # activity_regularizer=l2(self.l2_regularizer)
        ))

        # Reverse the operation for consistency
        # self.hidden_neurons_.pop(0)

        self.mod = model

    def call(self, x):
        return self.mod(x)


class AutoencoderAD():
    def __init__(self, auto):
        self.auto = auto

    def fit(self, x_train):
        self.auto.compile(optimizer=self.auto.optimizer, loss=self.auto.loss)
        history = self.auto.fit(x_train, x_train,
                                epochs=self.auto.epochs,
                                batch_size=self.auto.batch_size,
                                shuffle=True,
                                validation_split=self.auto.validation_size,
                                #validation_data=(x_test, x_test)
                                verbose=0
                                )
        return self

    def __str__(self):
        return 'AutoAD'

    def __repr__(self):
        return 'AutoAD'

    def predict(self, x_test):
        return self.auto.predict(x_test)
