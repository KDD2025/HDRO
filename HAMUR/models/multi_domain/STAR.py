import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2


def activation_layer(activation):
    if isinstance(activation, str):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer


class STAR(Layer):

    def __init__(self, hidden_units, num_domains, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.num_domains = num_domains
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(STAR, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        ## 共享FCN权重
        self.shared_kernels = [self.add_weight(name='shared_kernel_' + str(i),
                                               shape=(
                                                   hidden_units[i], hidden_units[i + 1]),
                                               initializer=glorot_normal(
                                                   seed=self.seed),
                                               regularizer=l2(self.l2_reg),
                                               trainable=True) for i in range(len(self.hidden_units))]

        self.shared_bias = [self.add_weight(name='shared_bias_' + str(i),
                                            shape=(self.hidden_units[i],),
                                            initializer=Zeros(),
                                            trainable=True) for i in range(len(self.hidden_units))]
        ## domain-specific 权重
        self.domain_kernels = [[self.add_weight(name='domain_kernel_' + str(index) + str(i),
                                                shape=(
                                                    hidden_units[i], hidden_units[i + 1]),
                                                initializer=glorot_normal(
                                                    seed=self.seed),
                                                regularizer=l2(self.l2_reg),
                                                trainable=True) for i in range(len(self.hidden_units))] for index in
                               range(self.num_domains)]

        self.domain_bias = [[self.add_weight(name='domain_bias_' + str(index) + str(i),
                                             shape=(self.hidden_units[i],),
                                             initializer=Zeros(),
                                             trainable=True) for i in range(len(self.hidden_units))] for index in
                            range(self.num_domains)]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(STAR, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, domain_indicator, training=None, **kwargs):
        deep_input = inputs
        output_list = [inputs] * self.num_domains
        for i in range(len(self.hidden_units)):
            for j in range(self.num_domains):
                # 网络的权重由共享FCN和其domain-specific FCN的权重共同决定
                output_list[j] = tf.nn.bias_add(tf.tensordot(
                    output_list[j], self.shared_kernels[i] * self.domain_kernels[j][i], axes=(-1, 0)),
                    self.shared_bias[i] + self.domain_bias[j][i])

                try:
                    output_list[j] = self.activation_layers[i](output_list[j], training=training)
                except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                    print("make sure the activation function use training flag properly", e)
                    output_list[j] = self.activation_layers[i](output_list[j])
        output = tf.reduce_sum(tf.stack(output_list, axis=1) * tf.expand_dims(domain_indicator, axis=-1), axis=1)

        return output

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(STAR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

