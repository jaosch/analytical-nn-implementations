import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

def fun(x):
    pow_3 = tf.math.pow(x, 3)

    _t = tf.gather(pow_3, [0], axis=1) * tf.gather(pow_3, [1], axis=1)
    y = tf.concat([_t, _t], axis=1)
    return y

class MLP(layers.Layer):
    ''' A feed-forward neural network '''
    def __init__(self, units, activation):
        super().__init__()
        self.ls = []
        for (u, a) in zip(units, activation):
            self.ls += [layers.Dense(u, a)]

    def call(self, x):    
        for l in self.ls:
            x = l(x)
        return x
    
class Gradient(layers.Layer):
    ''' Computes gradients using automatic differentation '''
    def __init__(self, l):
        super().__init__()
        self.l = l
    
    def call(self, x):
        with tf.GradientTape() as g0:
            g0.watch(x)
            with tf.GradientTape() as g1:
                g1.watch(x)
                y = self.l(x)
            dy = g1.batch_jacobian(y, x)
            ddy = g0.batch_jacobian(dy, x)
        return y, dy, ddy
    
def build(**kwargs):
    ''' Build and return tensorflow model '''
    x = tf.keras.Input(shape=(2,))
    mlp = MLP(**kwargs)
    y, dy, ddy = Gradient(mlp)(x)
    model = tf.keras.Model(inputs = [x], outputs = [y, dy, ddy])
    model.compile('adam', 'mse')
    return model

def main():
    # generate data
    x = tf.constant(np.linspace([-2, -2], [2, 2], 100), dtype='float32')
    grad = Gradient(fun)
    y, dy, ddy = grad(x)

    # build tensorflow model
    m = build(
        units = [16, 16, 2],
        activation = ['softplus', 'softplus', 'linear'],
    )

    # model calibration
    m.fit([x], [y, dy, ddy], epochs=5000, verbose=2)

    # save weights to txt-file
    with open('weights.txt', 'w') as f:
        for weights in m.get_weights():
            weights = weights.reshape(-1)
            f.write(' '.join(str(val) for val in weights) + '\n')

    # evaluation
    y_pred, dy_pred, ddy_pred = m(x)

    fig1, axs = plt.subplots(1, 2, figsize=(8, 8))
    for i, ax in enumerate(axs.reshape(-1)):
        ax.plot(y[:, i], y_pred[:, i], 'b.')
        ax.set_xlabel(f'Data: y{i}')
        ax.set_ylabel(f'Prediction: y{i}')
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        ax.plot([-1e8, 1e8], [-1e8, 1e8], 'k--')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
    fig1.suptitle('Correlation of function values', fontsize=16,  y=.995)
    plt.tight_layout()

    fig2, axs = plt.subplots(2, 2, figsize=(8, 8))
    for i in range(2):
        for j in range(2):
            axs[i, j].plot(dy[:, i, j], dy_pred[:, i, j], 'b.')
            axs[i, j].set_xlabel(f'Data: dy{i}{j}')
            axs[i, j].set_ylabel(f'Prediction: dy{i}{j}')
            x_lim = axs[i, j].get_xlim()
            y_lim = axs[i, j].get_ylim()
            axs[i, j].plot([-1e8, 1e8], [-1e8, 1e8], 'k--')
            axs[i, j].set_xlim(x_lim)
            axs[i, j].set_ylim(y_lim)
    fig2.suptitle('Correlation of derivatives', fontsize=16,  y=.995)
    plt.tight_layout()

    fig3, axs = plt.subplots(4, 2, figsize=(6, 12))
    for i in range(2):
        l = 0
        for j in range(2):
            for k in range(2):
                axs[l, i].plot(ddy[:, i, j, k], ddy_pred[:, i, j, k], 'b.')
                axs[l, i].set_xlabel(f'Data: ddy{i}{j}{k}')
                axs[l, i].set_ylabel(f'Prediction: ddy{i}{j}{k}')
                x_lim = axs[l, i].get_xlim()
                y_lim = axs[l, i].get_ylim()
                axs[l, i].plot([-1e8, 1e8], [-1e8, 1e8], 'k--')
                axs[l, i].set_xlim(x_lim)
                axs[l, i].set_ylim(y_lim)
                l += 1
    fig3.suptitle('Correlation of second derivatives', fontsize=16,  y=.995)
    plt.tight_layout()
    plt.show()

main()


