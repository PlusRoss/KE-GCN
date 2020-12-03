from inits import *
import tensorflow as tf
import sys

flags = tf.app.flags
FLAGS = flags.FLAGS
EOS = 1e-9

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def conjugate(x):
    x_shape = tf.shape(x)
    x = tf.reshape(x, [-1, 2])
    return tf.reshape(tf.stack([x[:, 0], -x[:, 1]],axis=1), x_shape)


def inverse_complex(x):
    x_shape = tf.shape(x)
    x = tf.reshape(x, [-1, 2])
    x = x / (tf.reduce_sum(x**2, 1, keepdims=True) + EOS)
    return tf.reshape(tf.stack([x[:, 0], -x[:, 1]],axis=1), x_shape)


def multiply_complex(x, y):
    result_real = x[:, ::2] * y[:, ::2] - x[:, 1::2] * y[:, 1::2]
    result_img = x[:, 1::2] * y[:, ::2] + x[:, ::2] * y[:, 1::2]
    return tf.reshape(tf.stack([result_real, result_img],axis=2), [tf.shape(x)[0], -1])


def multiply_quater(x, y):
    a, b, c, d = x[:, ::4], x[:, 1::4], x[:, 2::4], x[:, 3::4]
    p, q, u, v = y[:, ::4], y[:, 1::4], y[:, 2::4], y[:, 3::4]
    result_real = a*p - b*q - c*u - d*v
    result_i = a*q + b*p + c*v -d*u
    result_j = a*u - b*v + c*p + d*q
    result_k = a*v + b*u - c*q + d*p
    return tf.reshape(tf.stack([result_real, result_i, result_j, result_k],axis=2), [tf.shape(x)[0], -1])


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class InitLayer(Layer):
    '''
    Initialize Entity and Relation embedding
    '''

    def __init__(self, input_dim, output_dim, placeholders,
                 sparse_inputs=False, embed=None, dataset=None,
                 featureless=False, init=[glorot, glorot], **kwargs):
        super(InitLayer, self).__init__(**kwargs)

        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.embed = embed
        self.input_dim_ent, self.input_dim_rel = input_dim
        self.output_dim_ent, self.output_dim_rel = output_dim
        self.dataset = dataset
        self.loss = 0

        with tf.variable_scope(self.name + '_vars'):
            # vars: embedding of nodes
            for i in range(len(self.support)):
                if self.embed == "text":
                    self.vars['ent_embeds_' + str(i)] = text_embed(self.dataset, [self.input_dim_ent, self.output_dim_ent],
                                                            name='ent_embeds_' + str(i))
                else:
                    self.vars['ent_embeds_' + str(i)] = init[0]([self.input_dim_ent, self.output_dim_ent],
                                                            name='ent_embeds_' + str(i))
                self.vars['rel_embeds_' + str(i)] = init[1]([self.input_dim_rel, self.output_dim_rel],
                                                        name='rel_embeds_' + str(i))
                self.loss += tf.nn.l2_loss(self.vars['ent_embeds_' + str(i)])/self.input_dim_ent
                self.loss += tf.nn.l2_loss(self.vars['rel_embeds_' + str(i)])/self.input_dim_rel

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_ent', outputs[0])
                tf.summary.histogram(self.name + '/outputs_rel', outputs[1])
            return outputs

    def _call(self, inputs):
        ent_supports = list()
        rel_supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_ent = dot(x[0], self.vars['ent_embeds_' + str(i)], sparse=self.sparse_inputs)
            else:
                pre_ent = self.vars['ent_embeds_' + str(i)]
            pre_rel = self.vars['rel_embeds_' + str(i)]

            ent_supports.append(pre_ent)
            rel_supports.append(pre_rel)

        output_ent = tf.add_n(ent_supports)
        output_rel = tf.add_n(rel_supports)

        return output_ent, output_rel


class AutoRelGraphConvolution(Layer):
    '''
    Graph convolution layer based on auto-diff operation
    Currently supported KE methods: TransE, TransH, TransD, RotatE, QuatE, DistMult
    '''

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., alpha=0.5, beta=None, mode="None",
                 sparse_inputs=False, act=tf.nn.relu, bias=False, dataset=None, embed=None,
                 featureless=True, transform=False, attention=False, init=[glorot, glorot],
                 rel_update=True, truncate_ent=False, **kwargs):
        super(AutoRelGraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.transform = transform
        self.embed = embed
        self.input_dim_ent, self.input_dim_rel = input_dim
        self.output_dim_ent, self.output_dim_rel = output_dim
        self.mode = mode # None, TransE
        self.alpha = alpha
        if beta is None:
            self.beta = alpha
        else:
            self.beta = beta
        self.rel_update = rel_update

        self.normalize = False
        self.highway = False
        self.loss = 0
        self.truncate_ent = truncate_ent

        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            # vars: embedding of nodes
            for i in range(len(self.support)):
                if self.transform:
                    self.vars['ent_weights_' + str(i)] = init[0]([self.input_dim_ent, self.output_dim_ent],
                                                            name='ent_weights_' + str(i))
                    self.loss += tf.nn.l2_loss(self.vars['ent_weights_' + str(i)])

            if self.bias:
                self.vars['ent_bias'] = zeros([self.output_dim_ent], name='ent_bias')
                self.vars['rel_bias'] = zeros([self.output_dim_rel], name='rel_bias')

        if self.highway:
            self.kernel_gate_ent = glorot([self.output_dim_ent, self.output_dim_ent])
            self.bias_gate_ent = zeros([self.output_dim_ent])

        if self.logging:
            self._log_vars()

    def _message(self, ent_emb, rel_emb, nei_array, mode="None"):
        ent_head = tf.gather(ent_emb, nei_array[:,0])
        ent_tail = tf.gather(ent_emb, nei_array[:,2])
        rel = tf.gather(rel_emb, nei_array[:,1])
        if mode == "None":
            loss = - tf.reduce_sum((ent_head - ent_tail)**2)
        elif mode == "TransE":
            loss = - tf.reduce_sum((ent_head + rel - ent_tail)**2)
        elif mode == "TransH":
            rel_dim = tf.cast(tf.shape(rel)[1]/2, tf.int32)
            rel_1, rel_2 = rel[:, :rel_dim], rel[:, rel_dim:]
            rel_2_norm = tf.math.l2_normalize(rel_2, axis=1)
            ent_head_new = ent_head - tf.reduce_sum(ent_head * rel_2_norm, 1, True)/10. * rel_2_norm
            ent_tail_new = ent_tail - tf.reduce_sum(ent_tail * rel_2_norm, 1, True)/10. * rel_2_norm
            loss = - tf.reduce_sum((ent_head_new + rel_1 - ent_tail_new)**2)
        elif mode == "TransD":
            rel_dim = tf.cast(tf.shape(rel_emb)[1]/2, tf.int32)
            rel_1, rel_2 = rel[:, :rel_dim], rel[:, rel_dim:]
            ent_dim = tf.cast(tf.shape(ent_emb)[1]/2, tf.int32)
            ent_head_1, ent_head_2 = ent_head[:, :ent_dim], ent_head[:, ent_dim:]
            ent_tail_1, ent_tail_2 = ent_tail[:, :ent_dim], ent_tail[:, ent_dim:]
            ent_head_new = ent_head_1 - tf.reduce_sum(ent_head_1 * ent_head_2, 1, True)/10. * rel_2
            ent_tail_new = ent_tail_1 - tf.reduce_sum(ent_tail_1 * ent_tail_2, 1, True)/10. * rel_2
            loss = - tf.reduce_sum((ent_head_new + rel_1 - ent_tail_new)**2)
        elif mode == "DistMult":
            loss = - tf.reduce_sum((ent_head * rel - ent_tail)**2)
        elif mode == "RotatE":
            loss = -tf.reduce_sum((multiply_complex(ent_head, rel) - ent_tail)**2)
        elif mode == "QuatE":
            loss = -tf.reduce_sum((multiply_quater(ent_head, rel) - ent_tail)**2)
        ent_message, rel_message = tf.gradients(loss, [ent_emb, rel_emb])
        if mode == "None" or self.rel_update == False:
            rel_message = None
        return ent_message, rel_message, loss

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_ent', outputs[0])
                tf.summary.histogram(self.name + '/outputs_rel', outputs[1])
            return outputs

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.dropout:
            if self.sparse_inputs:
                x[0] = sparse_dropout(x[0], 1-self.dropout, self.num_features_nonzero)
                x[1] = sparse_dropout(x[1], 1-self.dropout, self.num_features_nonzero)
            else:
                x[0] = tf.nn.dropout(x[0], 1-self.dropout)
                x[1] = tf.nn.dropout(x[1], 1-self.dropout)

        # convolve
        ent_supports = list()
        rel_supports = list()
        for i in range(len(self.support)):
            pre_ent = x[0]
            pre_rel = x[1]

            # normalize the relation embedding in RotatE and QuatE
            if self.mode == "RotatE":
                rel_shape = pre_rel.shape
                pre_rel = tf.math.l2_normalize(tf.reshape(pre_rel, [-1, 2]), axis=1)
                pre_rel = tf.reshape(pre_rel, rel_shape)
            elif self.mode == "QuatE":
                rel_shape = pre_rel.shape
                pre_rel = tf.math.l2_normalize(tf.reshape(pre_rel, [-1, 4]), axis=1)
                pre_rel = tf.reshape(pre_rel, rel_shape)

            ent_invsum, rel_invsum, nei_array = self.support[i]
            ent_message, rel_message, loss = self._message(pre_ent, pre_rel, nei_array, self.mode)

            ent_update = ent_invsum * ent_message
            ent_support = pre_ent + self.alpha * ent_update

            if rel_message is not None:
                rel_update = rel_invsum * rel_message
                rel_support = pre_rel + self.beta * rel_update
            else:
                rel_support = pre_rel

            if self.transform:
                if self.truncate_ent and self.mode == "TransD":
                    ent_dim = tf.cast(tf.shape(ent_support)[1]/2, tf.int32)
                    ent_support = dot(ent_support[:, :ent_dim], self.vars['ent_weights_' + str(i)][:ent_dim])
                else:
                    ent_support = dot(ent_support, self.vars['ent_weights_' + str(i)])

            ent_supports.append(ent_support)
            rel_supports.append(rel_support)

        output_ent = tf.add_n(ent_supports)
        output_rel = tf.add_n(rel_supports)

        if self.bias:
            output_ent += self.vars['ent_bias']
            output_rel += self.vars['rel_bias']

        output_ent = self.act(output_ent)
        output_rel = self.act(output_rel)

        if self.normalize:
            output_ent = tf.math.l2_normalize(output_ent, axis=1)
            output_rel = tf.math.l2_normalize(output_rel, axis=1)

        return output_ent, output_rel
