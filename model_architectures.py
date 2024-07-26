import os

import tensorflow as tf
import keras
import keras.layers as kl
from keras.layers import Conv1D, MaxPooling1D, MultiHeadAttention, Cropping1D
from keras.layers import Layer, Activation, Dense, Flatten
from keras.layers import Dropout, BatchNormalization, LayerNormalization
from keras import models
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import backend as K
from keras_nlp.layers import RotaryEmbedding, PositionEmbedding, SinePositionEncoding

import evoaug_tf
from evoaug_tf import evoaug, augment

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error


class BaseNN:
    def __init__(self, params=None, num_inputs=1000, num_outputs=18):
        self.input_shape = (num_inputs, 4)
        self.num_outputs = num_outputs
        self.initialize_model(params)

    def initialize_model(self, params=None):
        raise NotImplementedError("Subclasses should implement this method")

    def lr_schedule(self, epoch, lr):
        return lr

    def _exp_decay(self, epoch, lr):
        return float(lr * tf.math.exp(-self.params['decay_rate']))

    def _cosine_decay(self, epoch, lr):
        # Cosine decay with warm restart
        initial_lr = self.params['lr'] # Initial learning rate
        min_lr = 1e-5    # Minimum learning rate
        decay_epochs = 1  # Number of epochs for each decay cycle
        warmup_epochs = 5  # Number of warmup epochs at the start of each cycle

        # Calculate the current cycle
        cycle = np.floor(1 + epoch / decay_epochs)
        # Calculate where we are in the current cycle
        x = np.abs(epoch / decay_epochs - cycle + 1)

        # Warmup phase
        if epoch % decay_epochs < warmup_epochs:
            return initial_lr * (epoch % decay_epochs) / warmup_epochs

        # Cosine decay phase
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * x))

    def train_model(self, X_train, Y_train, X_val, Y_val,
                    batch_size, epochs, early_stop=None, learning_rate=0.01):
        callbacks = []
        if early_stop is not None:
            callbacks.append(EarlyStopping(patience=early_stop,
                                           monitor="val_loss",
                                           restore_best_weights=True))
        # Add learning rate scheduler
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        callbacks.append(lr_scheduler)

        history = self.model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_data=(X_val, Y_val),
            callbacks=callbacks)
        return history

    def evaluate_model(self, X, Y, split):
        pred = self.model.predict(X)
        Y = Y.reshape(-1)
        pred = pred.reshape(-1)
        mse = mean_squared_error(Y, pred)
        pcc = pearsonr(Y, pred)[0]
        scc = spearmanr(Y, pred)[0]
        print(f'{split}: MSE = {mean_squared_error(Y, pred):.4f}')
        print(f'{split}: PCC = {pearsonr(Y, pred)[0]:.4f}')
        print(f'{split}: SCC = {spearmanr(Y, pred)[0]:.4f}')
        return {'MSE': mse, 'PCC': pcc, 'SCC': scc}

    def save_model(self, filetag='model', save_folder=None, format='h5'):
        if save_folder is None:
            raise NameError('save_folder must have a designated path')
        self.model.save(os.path.join(save_folder, f'{filetag}.{format}'))


class NormalNN(BaseNN):
    def lr_schedule(self, epoch, lr):
        new_lr = self._exp_decay(epoch, lr)
        # new_lr = self._cosine_decay()
        return new_lr

    def initialize_model(self, params):
        self.params = params

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
        ilayer += 1

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
        for _ in range(0, params['num_dense_layers']):
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
            loss='mean_squared_error')
            # metrics=[SpearmanCorrelation()])


class MHA_NN(BaseNN):
    def lr_schedule(self, epoch, lr):
        return self._exp_decay(epoch, lr)

    def initialize_model(self, params):
        self.params = params

        # Input layer
        x_input = kl.Input(shape=self.input_shape)
        x = x_input

        ilayer = 1
        x = Conv1D(
            filters=params[f'num_filters{ilayer}'],
            kernel_size=params[f'kernel_size{ilayer}'],
            padding=params['padding'],
            name=f'layer{ilayer}_conv1d')(x)
        x = BatchNormalization()(x)
        x = Activation(params[f'activation{ilayer}'])(x)
        x = MaxPooling1D(params[f'max_pool{ilayer}'])(x)
        ilayer += 1

        x = LayerNormalization()(x)
        x = RotaryEmbedding()(x)
        # x = PositionEmbeddin(sequence_length=)(x)
        attention_output = MultiHeadAttention(
            num_heads=params['num_heads'],
            key_dim=params['key_dim'],
            dropout=params['dropout_MHA'],
            name='multihead_attention')(x, x)

        x = kl.Add()([x, attention_output])  # Residual connection
        # x = Conv1D(
        #     filters=params['mha_output_dim'],
        #     kernel_size=1,
        #     name='adjust_output_shape_conv')(x)
        # x = BatchNormalization()(x)
        # x = Activation(params[f'activation{ilayer}'])(x)
        x = MaxPooling1D(params[f'max_pool{ilayer}'])(x)
        ilayer += 1

        x = Flatten()(x)

        # Dense layers
        x = Dense(
            params[f'dense_neurons{ilayer}'],
            name=f'layer{ilayer}_dense')(x)
        x = BatchNormalization()(x)
        x = Activation(params[f'activation{ilayer}'])(x)
        x = Dropout(params[f'dropout_prob{ilayer}'])(x)
        ilayer += 1

        x = Dense(
            params[f'dense_neurons{ilayer}'],
            name=f'layer{ilayer}_dense')(x)
        x = BatchNormalization()(x)
        x = Activation(params[f'activation{ilayer}'])(x)
        x = Dropout(params[f'dropout_prob{ilayer}'])(x)
        ilayer += 1

        # Flatten and dense layers
        bottleneck = x

        # Output
        outputs = Dense(
            self.num_outputs,
            activation='linear',
            name='dense_output')(bottleneck)

        self.model = Model([x_input], outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=params['lr']),
            loss='mean_squared_error')


class MHA_NN0(BaseNN):
    def initialize_model(self, params):
        # Input layer
        x_input = kl.Input(shape=self.input_shape)
        x = x_input

        ilayer = 1
        x = Conv1D(
            filters=params[f'num_filters{ilayer}'],
            kernel_size=params[f'kernel_size{ilayer}'],
            padding=params['padding'],
            name=f'layer{ilayer}_conv1d')(x)
        x = BatchNormalization()(x)
        x = Activation(params[f'activation{ilayer}'])(x)
        x = MaxPooling1D(params[f'max_pool{ilayer}'])(x)
        ilayer += 1

        x = LayerNormalization()(x)
        attention_output = MultiHeadAttention(
            num_heads=params['num_heads'],
            key_dim=params['key_dim'],
            dropout=params['dropout_MHA'],
            output_shape=params['num_outputs_MHA'],
            name='multihead_attention')(x, x)
        x = kl.Add()([x, attention_output])  # Residual connection
        # x = BatchNormalization()(x)
        # x = Activation(params[f'activation{ilayer}'])(x)
        ilayer += 1

        # Feedforward network
        x = LayerNormalization()(x)
        x = Conv1D(
            filters=params[f'num_filters{ilayer}'],
            kernel_size=1,
            padding=params['padding'],
            name='ff_conv1')(x)
        x = Dropout(params['dropout_ff'])(x)
        x = Activation('relu')(x)
        ilayer += 1

        # x = Conv1D(
        #     filters=params[f'num_filters{ilayer}'],
        #     kernel_size=1,
        #     padding=params['padding'],
        #     name='ff_conv2')(x)
        # x = Dropout(params['dropout_ff'])(x)
        # ilayer += 1

        # Cropping layer
        # x = Cropping1D(cropping=(320, 320))(x)

        # Pointwise convolution
        x = Conv1D(
            filters=params[f'num_filters{ilayer}'],
            kernel_size=1,
            padding='same',
            name='pointwise_conv')(x)
        x = BatchNormalization()(x)
        x = Activation('gelu')(x)
        x = Dropout(params['dropout_pw'])(x)
        ilayer += 1

        # Output head
        x = Conv1D(
            filters=self.num_outputs,
            kernel_size=1,
            padding='same',
            name='output_conv')(x)
        x = Activation('gelu')(x)

        # Flatten and dense layers
        bottleneck = Flatten()(x)

        # Output
        outputs = Dense(
            self.num_outputs,
            activation='linear',
            name='dense_output')(bottleneck)

        self.model = Model([x_input], outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=params['lr']),
            loss='mean_squared_error')


class EvoAug_NN(BaseNN):
    def __init__(self, params=None, num_inputs=1000, num_outputs=18):
        self.input_shape = (num_inputs, 4)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.initialize_model(params)

    def initialize_model(self, params):
        self.params = params

        augment_list = [
            augment.RandomRC(rc_prob=0.5),
            augment.RandomDeletion(delete_min=0, delete_max=20),
            augment.RandomTranslocation(shift_min=0, shift_max=20),
            augment.RandomNoise(noise_mean=0, noise_std=0.2),
            augment.RandomMutation(mutate_frac=0.05)
        ]

        self.model = evoaug.RobustModel(
            CNN,
            input_shape=(self.num_inputs, 4),
            augment_list=augment_list,
            max_augs_per_seq=2,
            hard_aug=True)
        # self.model = evoaug.RobustModel(
        #     self.NormalNN_fn,
        #     input_shape=(self.num_inputs, 4),
        #     augment_list=augment_list,
        #     max_augs_per_seq=2,
        #     hard_aug=True,
        #     num_outputs = self.num_outputs,
        #     params = params)
        optimizer = keras.optimizers.Adam(learning_rate=params['lr'])
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    @staticmethod
    def NormalNN_fn(input_shape, num_outputs, params):
        params = params
        # Input layer
        x_input = kl.Input(shape=input_shape)
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
        ilayer += 1

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
        for _ in range(0, params['num_dense_layers']):
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
            num_outputs,
            activation='linear',
            name='dense_output')(bottleneck)

        return Model([x_input], outputs)


def CNN(input_shape):
    def residual_block(input_layer, filter_size, activation='relu', dilated=5):
        factor = []
        base = 2
        for i in range(dilated):
            factor.append(base**i)

        num_filters = input_layer.shape.as_list()[-1]

        nn = keras.layers.Conv1D(filters=num_filters,
                                        kernel_size=filter_size,
                                        activation=None,
                                        use_bias=False,
                                        padding='same',
                                        dilation_rate=1,
                                        )(input_layer)
        nn = keras.layers.BatchNormalization()(nn)
        for f in factor:
            nn = keras.layers.Activation('relu')(nn)
            nn = keras.layers.Dropout(0.1)(nn)
            nn = keras.layers.Conv1D(filters=num_filters,
                                          kernel_size=filter_size,
                                          strides=1,
                                          activation=None,
                                          use_bias=False,
                                          padding='same',
                                          dilation_rate=f,
                                          )(nn)
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.add([input_layer, nn])
        return keras.layers.Activation(activation)(nn)

    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(196, kernel_size=19, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('exponential')(x)
    x = keras.layers.Dropout(0.1)(x)
    # x = residual_block(x, 3, activation='relu', dilated=3)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.MaxPooling1D(10)(x)

    x = keras.layers.Conv1D(256, kernel_size=7, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.MaxPooling1D(5)(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    outputs = keras.layers.Dense(14, activation='linear')(x)
    return keras.Model(inputs=inputs, outputs=outputs)
