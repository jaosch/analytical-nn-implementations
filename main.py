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

    def get_config(self):
        config = super().get_config()
        config.update({
            "l": self.l
        })
        return config


    def call(self, x):
        with tf.GradientTape() as g2:
            g2.watch(x)
            with tf.GradientTape() as g1:
                g1.watch(x)
                with tf.GradientTape() as g0:
                    g0.watch(x)
                    y = self.l(x)
                dy = g0.batch_jacobian(y, x)
            ddy = g1.batch_jacobian(dy, x)
        dddy = g2.batch_jacobian(ddy, x)
        return y, dy, ddy, dddy

def build(**kwargs):
    ''' Build and return tensorflow model '''
    x = tf.keras.Input(shape=(2,))
    mlp = MLP(**kwargs)
    y, dy, ddy, dddy = Gradient(mlp)(x)
    model = tf.keras.Model(inputs = [x], outputs = [y, dy, ddy, dddy])
    model.compile('adam', 'mse')
    return model

# Main training class
class Training:
    def __init__(self):
        self.x: tf.Tensor
        self.y: tf.Tensor
        self.dy: tf.Tensor
        self.ddy: tf.Tensor
        self.dddy: tf.Tensor

        self.model: tf.keras.Model

    def run(self):
        self._generate_data()
        self._build_model()
        self._train()
        self._evaluate()

    def _generate_data(self):
        self.x = tf.constant(np.linspace([-2, -2], [2, 2], 100), dtype='float32')
        grad = Gradient(fun)
        self.y, self.dy, self.ddy, self.dddy = grad(self.x)

        print('\nGenerated data with the following shapes:\n' + '-'*41)
        print(f'Input:\t\t{self.x.shape}')
        print(f'Values:\t\t{self.y.shape}')
        print(f'1st derivative:\t{self.dy.shape}')
        print(f'2nd derivative:\t{self.ddy.shape}')
        print(f'3rd derivative:\t{self.dddy.shape}')

    def _build_model(self):
        print('\nBuilding model: ...')

        self.model = build(
            units = [16, 16, 2],
            activation = ['softplus', 'softplus', 'linear'],
        )

        self.model.summary()

    def _train(self):
        print('\nTraining model: ...')

        self.model.fit([self.x],[self.y, self.dy, self.ddy, self.dddy],
            epochs=5000,
            verbose=0)

        self.model.save('tf_model.keras')

    def _evaluate(self):
        print('\nEvaluating model: ...')
        # save weights to txt-file
        with open('weights.txt', 'w') as f:
            for weights in self.model.get_weights():
                weights = weights.reshape(-1)
                f.write(' '.join(str(val) for val in weights) + '\n')

        # evaluation
        y_pred, dy_pred, ddy_pred, dddy_pred = self.model(self.x)

        fig1, axs = plt.subplots(1, 2, figsize=(8, 8))
        for i, ax in enumerate(axs.reshape(-1)):
            ax.plot(self.y[:, i], y_pred[:, i], 'b.')
            ax.set_xlabel(f'Data: y{i}')
            ax.set_ylabel(f'Prediction: y{i}')
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            ax.plot([-1e8, 1e8], [-1e8, 1e8], 'k--')
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
        fig1.suptitle('Correlation of function values', fontsize=16,  y=.995)
        plt.tight_layout()
        fig1.savefig("correlation_values.png", dpi=300)

        fig2, axs = plt.subplots(2, 2, figsize=(8, 8))
        for i in range(2):
            for j in range(2):
                axs[i, j].plot(self.dy[:, i, j], dy_pred[:, i, j], 'b.')
                axs[i, j].set_xlabel(f'Data: dy{i}{j}')
                axs[i, j].set_ylabel(f'Prediction: dy{i}{j}')
                x_lim = axs[i, j].get_xlim()
                y_lim = axs[i, j].get_ylim()
                axs[i, j].plot([-1e8, 1e8], [-1e8, 1e8], 'k--')
                axs[i, j].set_xlim(x_lim)
                axs[i, j].set_ylim(y_lim)
        fig2.suptitle('Correlation of derivatives', fontsize=16,  y=.995)
        plt.tight_layout()
        fig2.savefig("correlation_derivatives.png", dpi=300)

        fig3, axs = plt.subplots(4, 2, figsize=(6, 12))
        for i in range(2):
            l = 0
            for j in range(2):
                for k in range(2):
                    axs[l, i].plot(self.ddy[:, i, j, k], ddy_pred[:, i, j, k], 'b.')
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
        fig3.savefig("correlation_2nd_derivatives.png", dpi=300)

        fig3, axs = plt.subplots(8, 2, figsize=(6, 24))
        for i in range(2):
            m = 0
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        axs[m, i].plot(self.dddy[:, i, j, k, l], dddy_pred[:, i, j, k, l], 'b.')
                        axs[m, i].set_xlabel(f'Data: dddy{i}{j}{k}{l}')
                        axs[m, i].set_ylabel(f'Prediction: dddy{i}{j}{k}{l}')
                        x_lim = axs[l, i].get_xlim()
                        y_lim = axs[l, i].get_ylim()
                        axs[m, i].plot([-1e8, 1e8], [-1e8, 1e8], 'k--')
                        axs[m, i].set_xlim(x_lim)
                        axs[m, i].set_ylim(y_lim)
                        m += 1
        fig3.suptitle('Correlation of third derivatives', fontsize=16,  y=.995)
        plt.tight_layout()
        fig3.savefig("correlation_3rd_derivatives.png", dpi=300)


def main():
    training = Training();
    training.run()
    # # generate data
    # x = tf.constant(np.linspace([-2, -2], [2, 2], 100), dtype='float32')
    # grad = Gradient(fun)
    # y, dy, ddy = grad(x)

    # # build tensorflow model
    # m = build(
    #     units = [16, 16, 2],
    #     activation = ['softplus', 'softplus', 'linear'],
    # )

    # # model calibration
    # m.fit([x], [y, dy, ddy], epochs=5000, verbose=2)

    # # save weights to txt-file
    # with open('weights.txt', 'w') as f:
    #     for weights in m.get_weights():
    #         weights = weights.reshape(-1)
    #         f.write(' '.join(str(val) for val in weights) + '\n')

    # # evaluation
    # y_pred, dy_pred, ddy_pred = m(x)

    # fig1, axs = plt.subplots(1, 2, figsize=(8, 8))
    # for i, ax in enumerate(axs.reshape(-1)):
    #     ax.plot(y[:, i], y_pred[:, i], 'b.')
    #     ax.set_xlabel(f'Data: y{i}')
    #     ax.set_ylabel(f'Prediction: y{i}')
    #     x_lim = ax.get_xlim()
    #     y_lim = ax.get_ylim()
    #     ax.plot([-1e8, 1e8], [-1e8, 1e8], 'k--')
    #     ax.set_xlim(x_lim)
    #     ax.set_ylim(y_lim)
    # fig1.suptitle('Correlation of function values', fontsize=16,  y=.995)
    # plt.tight_layout()
    # fig1.savefig("correlation_values.png", dpi=300)

    # fig2, axs = plt.subplots(2, 2, figsize=(8, 8))
    # for i in range(2):
    #     for j in range(2):
    #         axs[i, j].plot(dy[:, i, j], dy_pred[:, i, j], 'b.')
    #         axs[i, j].set_xlabel(f'Data: dy{i}{j}')
    #         axs[i, j].set_ylabel(f'Prediction: dy{i}{j}')
    #         x_lim = axs[i, j].get_xlim()
    #         y_lim = axs[i, j].get_ylim()
    #         axs[i, j].plot([-1e8, 1e8], [-1e8, 1e8], 'k--')
    #         axs[i, j].set_xlim(x_lim)
    #         axs[i, j].set_ylim(y_lim)
    # fig2.suptitle('Correlation of derivatives', fontsize=16,  y=.995)
    # plt.tight_layout()
    # fig2.savefig("correlation_derivatives.png", dpi=300)

    # fig3, axs = plt.subplots(4, 2, figsize=(6, 12))
    # for i in range(2):
    #     l = 0
    #     for j in range(2):
    #         for k in range(2):
    #             axs[l, i].plot(ddy[:, i, j, k], ddy_pred[:, i, j, k], 'b.')
    #             axs[l, i].set_xlabel(f'Data: ddy{i}{j}{k}')
    #             axs[l, i].set_ylabel(f'Prediction: ddy{i}{j}{k}')
    #             x_lim = axs[l, i].get_xlim()
    #             y_lim = axs[l, i].get_ylim()
    #             axs[l, i].plot([-1e8, 1e8], [-1e8, 1e8], 'k--')
    #             axs[l, i].set_xlim(x_lim)
    #             axs[l, i].set_ylim(y_lim)
    #             l += 1
    # fig3.suptitle('Correlation of second derivatives', fontsize=16,  y=.995)
    # plt.tight_layout()
    # fig3.savefig("correlation_2nd_derivatives.png", dpi=300)


if __name__=="__main__":
    main()
