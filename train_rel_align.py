'''
code for relation alignment task
'''
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from metrics import *
from models import AutoRGCN_Align
import logging
import os

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'zh_en', 'Dataset name: zh_en, ja_en, fr_en')
flags.DEFINE_string('mode', 'None', 'KE method for GCN: TransE, TransH, TransD, DistMult, RotatE, QuatE')
flags.DEFINE_string('optim', 'Adam', 'Optimizer: GD, Adam')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gamma', 3.0, 'Hyper-parameter for margin based loss.')
flags.DEFINE_integer('num_negs', 5, 'Number of negative samples for each positive entity seed.')
flags.DEFINE_integer('num_rel_negs', 5, 'Number of negative samples for each positive relation seed.')
flags.DEFINE_float('alpha', 0.5, 'Weight of entity conv update.')
flags.DEFINE_float('beta', 0.5, 'Weight of relation conv update.')
flags.DEFINE_float('rel_weight', 0.5, 'weight of relation alignment loss.')
flags.DEFINE_integer('layer', 0, 'number of hidden layers')
flags.DEFINE_integer('dim', 200, 'hidden Dimension.')
flags.DEFINE_integer('seed', 3, 'Proportion of seeds, 3 means 30%')
flags.DEFINE_integer('rel_seed', 3, 'Proportion of relation seeds, 3 means 30%')
flags.DEFINE_boolean('auto', False, 'If true, uses autograd based GCN.')
flags.DEFINE_boolean('rel_align', False, 'If true, use relation alignment information.')
flags.DEFINE_boolean('rel_update', False, 'If true, use graph conv for rel update.')
flags.DEFINE_integer('randomseed', 12306, 'seed for randomness')
flags.DEFINE_boolean('valid', False, 'If true, split validation data.')
flags.DEFINE_boolean('save', False, 'If true, save the print')
flags.DEFINE_string('metric', "cityblock", 'metric for testing')
flags.DEFINE_string('loss_mode', "L1", 'mode for loss calculation')
flags.DEFINE_string('embed', "random", 'init embedding for entities') # random, text

np.random.seed(FLAGS.randomseed)
random.seed(FLAGS.randomseed)
tf.set_random_seed(FLAGS.randomseed)

if FLAGS.save:
    nsave = "log/{}/{}".format(FLAGS.dataset + "_rel", FLAGS.mode)
else:
    print("not saving file")
    nsave = "log/trash"
create_exp_dir(nsave)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p', filemode="w")
save_fname = 'alpha{}-beta{}-wrel{}-srel{}-layer{}-sdim{}-lr{}-seed{}'.format(
               FLAGS.alpha, FLAGS.beta, FLAGS.rel_weight, FLAGS.rel_seed, FLAGS.layer, FLAGS.dim,
               FLAGS.learning_rate, FLAGS.randomseed)

save_fname = save_fname + "-rel"
save_fname = "auto-" + save_fname
if not FLAGS.valid:
    save_fname = "test-" + save_fname
fh = logging.FileHandler(os.path.join(nsave, save_fname + ".txt"), "w")
# model_file = os.path.join(nsave, save_fname + ".pt")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.getLogger().setLevel(logging.INFO)


# Load data
adj, num_ent, train, test, valid, train_rel, test_rel, valid_rel = load_data_rel_align(FLAGS)
rel_num = np.max(adj[2][:, 1]) + 1
print("Entity num: ", num_ent)
print("Relation num: ", rel_num)
print("train rel num: ", len(train_rel))
print("rel weight: ", FLAGS.rel_weight)

# process graph to fit into later computation
support = [preprocess_adj(adj)]
num_supports = 1
model_func = AutoRGCN_Align

num_negs = FLAGS.num_negs
num_rel_negs = FLAGS.num_rel_negs

# Define placeholders
placeholders = {
    'features': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder_with_default(0, shape=())
}
placeholders['support'] = [[tf.placeholder(tf.float32, shape=[None, 1]),
                    tf.placeholder(tf.float32, shape=[None, 1]), \
                    tf.placeholder(tf.int32)] for _ in range(num_supports)]

# Create model
input_dim = [num_ent, rel_num]
hidden_dim = [FLAGS.dim, FLAGS.dim]
output_dim = [FLAGS.dim, FLAGS.dim]
if FLAGS.mode == "TransH":
    hidden_dim[1] *= 2
    output_dim[1] *= 2
elif FLAGS.mode == "TransD":
    hidden_dim[0] *= 2
    hidden_dim[1] *= 2
    output_dim[0] *= 2
    output_dim[1] *= 2
if len(train_rel) == 0:
    rel_align_loss = False
else:
    rel_align_loss = True
names_neg = [["left", "neg_right", "neg_left", "right"],
              ["left_rel", "neg_right_rel", "neg_left_rel", "right_rel"]]
model = model_func(placeholders, input_dim, hidden_dim, output_dim, dataset=FLAGS.dataset,
                    train_labels=train, REL=train_rel, mode=FLAGS.mode, embed=FLAGS.embed, alpha=FLAGS.alpha,
                    beta=FLAGS.beta, layer_num=FLAGS.layer, sparse_inputs=False, featureless=True,
                    logging=True, rel_update=FLAGS.rel_update, task="align",
                    rel_align_loss=rel_align_loss, names_neg=names_neg)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


num_labels = len(train)
left_labels = np.ones((num_labels, num_negs)) * (train[:, 0].reshape((num_labels, 1)))
left_labels = left_labels.reshape((num_labels * num_negs,))
right_labels = np.ones((num_labels, num_negs)) * (train[:, 1].reshape((num_labels, 1)))
right_labels = right_labels.reshape((num_labels * num_negs,))

if rel_align_loss:
    num_rel_labels = len(train_rel)
    left_rel_labels = np.ones((num_rel_labels, num_rel_negs)) * (train_rel[:, 0].reshape((num_rel_labels, 1)))
    left_rel_labels = left_rel_labels.reshape((num_rel_labels * num_rel_negs,))
    right_rel_labels = np.ones((num_rel_labels, num_rel_negs)) * (train_rel[:, 1].reshape((num_rel_labels, 1)))
    right_rel_labels = right_rel_labels.reshape((num_rel_labels * num_rel_negs,))

# Train model
for epoch in range(FLAGS.epochs):
    if epoch % 10 == 0:
        left_neg = np.random.choice(num_ent, num_labels * num_negs)
        right_neg = np.random.choice(num_ent, num_labels * num_negs)
        if rel_align_loss:
            left_rel_neg = np.random.choice(rel_num, num_rel_labels * num_rel_negs)
            right_rel_neg = np.random.choice(rel_num, num_rel_labels * num_rel_negs)
    feed_dict = construct_feed_dict(1.0, support, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    for i,labels in enumerate([left_labels,right_neg,left_neg,right_labels]):
        feed_dict.update({names_neg[0][i]+":0": labels})
    if rel_align_loss:
        for i,labels in enumerate([left_rel_labels,right_rel_neg,left_rel_neg,right_rel_labels]):
            feed_dict.update({names_neg[1][i]+":0": labels})
    # Training step
    if rel_align_loss:
        outs_se = sess.run([model.opt_op, model.loss, model.rel_loss], feed_dict=feed_dict)
    else:
        outs_se = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

    # Print results
    if epoch % 10 == 0:
        if rel_align_loss:
            logging.info("Epoch: {} train_loss_ent= {:.5f} train_loss_rel= {:.5f}".format(
                          epoch+1, outs_se[1], outs_se[2]))
        else:
            logging.info("Epoch: {} train_loss= {:.5f}".format(
                          epoch+1, outs_se[1]))

    if epoch % 100 == 0 and valid is not None:
        output_embeddings = sess.run(model.outputs, feed_dict=feed_dict)
        logging.info("Entity Results: ")
        get_align(output_embeddings[0], valid, logging, FLAGS.metric)
        logging.info("Relation Results: ")
        get_align(output_embeddings[1], valid_rel, logging, FLAGS.metric)

    if epoch % 2000 == 0 and epoch > 0 and valid is None:
        output_embeddings = sess.run(model.outputs, feed_dict=feed_dict)
        logging.info("Entity Results: ")
        get_align(output_embeddings[0], test, logging, FLAGS.metric)
        logging.info("Relation Results: ")
        get_align(output_embeddings[1], test_rel, logging, FLAGS.metric)

print("Optimization Finished!")

# Testing
if valid is not None:
    exit()

feed_dict = construct_feed_dict(1.0, support, placeholders)
output_embeddings = sess.run(model.outputs, feed_dict=feed_dict)
logging.info("Entity Results: ")
get_align(output_embeddings[0], test, logging, FLAGS.metric)
logging.info("Relation Results: ")
get_align(output_embeddings[1], test_rel, logging, FLAGS.metric)
