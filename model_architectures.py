import os

import tensorflow as tf

import keras
import keras.layers as kl
from keras.layers import Dense, BatchNormalization, InputLayer, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D, MaxPooling2D
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras import models
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error


class BaseNN:
    def __init__(self, params=None):
        self.input_shape = (1001, 4)
        self.num_outputs = 18
        self.initialize_model(params)

    def initialize_model(self, params=None):
        raise NotImplementedError("Subclasses should implement this method")

    def train_model(self, X_train, Y_train, X_val, Y_val,
                    batch_size, epochs, early_stop=None):
        if early_stop is not None:
            callbacks = [EarlyStopping(patience='early_stop',
                                       monitor="val_loss",
                                       restore_best_weights=True)]
        else:
            callbacks = None

        history = self.model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, Y_val),
            callbacks=callbacks)
        return history

    def evaluate_model(self, X, Y, split):
        pred = self.model.predict(X)
        print(f'{split}: MSE = {mean_squared_error(Y, pred.squeeze()):.2f}')
        print(f'{split}: PCC = {pearsonr(Y, pred.squeeze()):.2f}')
        print(f'{split}: SCC = {spearmanr(Y, pred.squeeze()):.2f}')

    def save_model(self, filetag='model', save_folder=None, format='h5'):
        if save_folder is None:
            raise NameError('save_folder must have a designated path')
        self.model.save(os.path.join(save_folder, f'{filetag}.{format}'))

    @staticmethod
    def spearman_correlation(y_true, y_pred):
        return (tf.py_function(spearmanr,
                [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)],
                Tout = tf.float32))


class SimpleNN(BaseNN):
    def initialize_model(self, params=None):
        # Input layer
        x_input = kl.Input(shape=self.input_shape)
        x = x_input

        # Body
        x = Conv1D(
            filters=128,
            kernel_size=7,
            padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=4)(x)

        x = Flatten()(x)

        # Dense
        x = Dense(256, activation='relu')(x)
        outputs = Dense(self. num_outputs, activation='linear')(x)

        self.model = Model([x_input], outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=0.005),
            # optimizer=SGD(learning_rate=0.005),
            loss='mean_squared_error',
            metrics=[self.spearman_correlation])


class NormalNN(BaseNN):
    def initialize_model(self, params):
        # Input layer
        x_input = kl.Input(shape=self.input_shape)
        x = x_input

        # Body
        ilayer = 1
        x = Conv1D(
            filters=params[f'num_filters{ilayer}'],
            kernel_size=params[f'kernel_size{ilayer}'],
            padding=params['padding'],
            name=f'layer{ilayer}_conv1d')(x)
        x = BatchNormalization()(x)
        x = Activation(params[f'activation{ilayer}'])(x)
        x = MaxPooling1D(params[f'max_pool{ilayer}'])(x)

        for _ in range(1, params['num_conv_layers']):
            x = Conv1D(
                filters=params[f'num_filters{ilayer}'],
                kernel_size=params[f'kernel_size{ilayer}'],
                padding=params['padding'],
                name=f'layer{ilayer}_conv1d')(x)
            x = BatchNormalization()(x)
            x = Activation(params[f'activation{ilayer}'])(x)
            x = MaxPooling1D(params[f'max_pool{ilayer}'])(x)
            ilayer += 1

        x = Flatten()(x)

        # Dense layers
        for _ in range(0, params['n_dense_layers']):
            x = Dense(
                params[f'dense_neurons{ilayer}'],
                name=f'layer{ilayer}_dense')(x)
            x = BatchNormalization()(x)
            x = Activation(params[f'activation{ilayer}'])(x)
            x = Dropout(params[f'dropout_prob{ilayer}'])(x)
            ilayer += 1

        bottleneck = x

        # Output
        outputs = Dense(
            self.num_outputs,
            activation='linear',
            name='dense_output')(bottleneck)

        self.model = Model([x_input], outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=params['lr']),
            loss='mean_squared_error',
            metrics=[self.spearman_correlation])
