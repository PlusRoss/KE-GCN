from layers import *
from metrics import align_loss, class_loss, label_loss
from inits import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', "task", "rel_update"}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging
        task = kwargs.get('task')
        self.task = task
        rel_update = kwargs.get('rel_update')
        self.rel_update = rel_update

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)



class AutoRGCN_Align(Model):
    def __init__(self, placeholders, input_dim, hidden_dim, output_dim, train_labels, dataset, REL=None,
                 mode="None", embed="random", alpha=0.5, beta=0.5, layer_num=0, sparse_inputs=False,
                 featureless=True, rel_align_loss=False, names_neg=None, **kwargs):
        super(AutoRGCN_Align, self).__init__(**kwargs)

        print("####### Model: AutoRGCN_Align #########")

        # inputs for layers
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.placeholders = placeholders
        self.train_labels = train_labels
        self.REL = REL
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.rel_align_loss = rel_align_loss
        self.mode = mode
        self.embed = embed
        self.alpha = alpha
        self.beta = beta
        self.layer_num = layer_num
        self.dataset = dataset
        self.names_neg = names_neg

        if FLAGS.optim == "GD":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        elif FLAGS.optim == "Adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        else:
            exit("Wrong optimizer")

        self.build()

    def _loss(self):
        if self.task == "align":
            # knowledge graph alignment
            self.align_loss = align_loss(self.outputs[0], self.train_labels, FLAGS.gamma, FLAGS.num_negs, self.names_neg[0], FLAGS.loss_mode)
            if not self.rel_align_loss:
                self.loss += self.align_loss
            else:
                self.rel_loss = rel_align_loss(self.outputs[1], self.REL, FLAGS.gamma, FLAGS.num_rel_negs, self.names_neg[1], FLAGS.loss_mode)
                self.loss += self.align_loss * (1-FLAGS.rel_weight) + self.rel_loss * FLAGS.rel_weight
        elif self.task == "class":
            # multi-class classification
            self.loss += class_loss(self.outputs[0], self.train_labels[0], self.train_labels[1])
        elif self.task == "label":
            # multi-label classification
            self.loss += label_loss(self.outputs[0], self.train_labels[0], self.train_labels[1])


    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            if self.task == "align":
                self._build_align()
            elif self.task in ["class", "label"]:
                self._build_class()
            else:
                exit("build error! wrong task!")

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        # output the last layer representation for prediction
        self.outputs = self.activations[-1]
        # For KG alignment task, if we use TransD, then only the first half embeddings
        # are used for candidate ranking
        if self.mode == "TransD" and self.task == "align":
            ent_dim = tf.cast(tf.shape(self.outputs[0])[1]/2, tf.int32)
            rel_dim = tf.cast(tf.shape(self.outputs[1])[1]/2, tf.int32)
            self.outputs = [self.outputs[0][:, :ent_dim], self.outputs[1][:, :rel_dim]]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _build_align(self):

        init_e = trunc_normal
        init_r = trunc_normal
        if self.mode == "DistMult":
            init_e = one_trunc_normal
            init_r = one_trunc_normal

        self.layers.append(InitLayer(input_dim=self.input_dim,
                                            output_dim=self.hidden_dim,
                                            placeholders=self.placeholders,
                                            embed=self.embed,
                                            dataset=self.dataset,
                                            featureless=self.featureless,
                                            sparse_inputs=self.sparse_inputs,
                                            init=[init_e, init_r],
                                            logging=self.logging))

        self.layers.append(AutoRelGraphConvolution(input_dim=self.hidden_dim,
                                            output_dim=self.hidden_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            alpha=self.alpha,
                                            beta=self.beta,
                                            mode=self.mode,
                                            bias=False,
                                            transform=False,
                                            init=[trunc_normal, trunc_normal],
                                            logging=self.logging))

        for i in range(self.layer_num):
            self.layers.append(AutoRelGraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=False,
                                                alpha=self.alpha,
                                                beta=self.beta,
                                                mode=self.mode,
                                                bias=False,
                                                transform=False,
                                                init=[trunc_normal, trunc_normal],
                                                logging=self.logging))

        self.layers.append(AutoRelGraphConvolution(input_dim=self.hidden_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            alpha=self.alpha,
                                            beta=self.beta,
                                            mode=self.mode,
                                            bias=False,
                                            transform=False,
                                            init=[trunc_normal, trunc_normal],
                                            logging=self.logging))

        for layer in self.layers:
            layer.rel_update = self.rel_update

    def _build_class(self):

        init_e = trunc_normal
        init_r = trunc_normal
        if self.mode == "DistMult":
            init_e = one_trunc_normal
            init_r = ones_static

        self.layers.append(InitLayer(input_dim=self.input_dim,
                                            output_dim=self.hidden_dim,
                                            placeholders=self.placeholders,
                                            embed=self.embed,
                                            dataset=self.dataset,
                                            featureless=self.featureless,
                                            sparse_inputs=self.sparse_inputs,
                                            init=[init_e, init_r],
                                            logging=self.logging))

        self.layers.append(AutoRelGraphConvolution(input_dim=self.hidden_dim,
                                            output_dim=self.hidden_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            alpha=self.alpha,
                                            beta=self.beta,
                                            mode=self.mode,
                                            bias=False,
                                            transform=False,
                                            init=[glorot, glorot],
                                            logging=self.logging))

        for i in range(self.layer_num):
            self.layers.append(AutoRelGraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=False,
                                                alpha=self.alpha,
                                                beta=self.beta,
                                                mode=self.mode,
                                                bias=False,
                                                transform=False,
                                                init=[glorot, glorot],
                                                logging=self.logging))

        self.layers.append(AutoRelGraphConvolution(input_dim=self.hidden_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x:x,
                                            dropout=False,
                                            alpha=self.alpha,
                                            beta=self.beta,
                                            mode=self.mode,
                                            bias=False,
                                            transform=True,
                                            truncate_ent=True,
                                            init=[glorot, glorot],
                                            logging=self.logging))

        for layer in self.layers:
            layer.rel_update = self.rel_update
