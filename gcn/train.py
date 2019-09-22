from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import networkx as nx
from sklearn.model_selection import KFold, StratifiedKFold
import datetime
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import softmax
from sklearn.metrics import classification_report

from gcn.utils import *
from gcn.models import GCN, MLP

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 2048, 'Number of units in GCN hidden layer 1.')
flags.DEFINE_integer('hidden2', 1024, 'Number of units in GCN hidden layer 2.')
flags.DEFINE_integer('dense1', 1024, 'Number of units in dense layer 1.')
flags.DEFINE_integer('dense2', 512, 'Number of units in dense layer 2.')
flags.DEFINE_integer('dense3', 256, 'Number of units in dense layer 3.')
flags.DEFINE_integer('dense4', 16, 'Number of units in dense layer 4.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

cutoff = 2
bert_dim = 1024  # each node, representing a token from ConceptNet, has got a feature vector X = H_0, encoded by BERT
node_num_dict = {1:15416, 2:134233, 3:236097, 5:288359, 10:297373}
node_num = node_num_dict[cutoff] 
G = nx.read_edgelist('/tf/gcn/gcn/SubG_' + str(cutoff) + '.tsv')
adj = nx.adjacency_matrix(G)
features = np.fromfile('/tf/gcn/gcn/node_feat_vec_H0_cutoff_' + str(cutoff) + '.txt', dtype=np.float32)
features = np.reshape(features, (node_num, bert_dim))
#print(features)

# Load data
list_1 = []
with open('/tf/gcn/gcn/HiEve_merged_' + str(cutoff) + '_set.tsv', 'r') as f_index:
    line = f_index.readline()
    while line:
        list_1.append(line[:-1])
        line = f_index.readline()
        
# triples are not changing
triples = []
with open('/tf/gcn/gcn/triples_46072_c.tsv', 'r') as f_data:
    line = f_data.readline()
    while line:
        line = line.split(',')
        triple_tmp = [list_1.index(line[0][2:-1]), list_1.index(line[1][2:-1]), line[2][1:-2]]
        triples.append(triple_tmp)
        
        line = f_data.readline()

triples = np.asarray(triples).astype(np.int32)
np.random.shuffle(triples)
id1_id2 = triples[:,[0,1]]
labels = triples[:,2]

kf = KFold(n_splits = 5, shuffle = True, random_state = seed)

X_train_cv = []
X_valid_cv = []
X_test_cv = []
y_train_cv = []
y_valid_cv = []
y_test_cv = []

for train_index, test_index in kf.split(id1_id2):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train_i, X_test_i = id1_id2[train_index], id1_id2[test_index]
    y_train_i, y_test_i = labels[train_index], labels[test_index]
    train_num = X_train_i.shape[0] - X_test_i.shape[0]    # test_num = valid_num
    X_train_cv.append(X_train_i[0:train_num,:])
    X_valid_cv.append(X_train_i[train_num:, :])
    X_test_cv.append(X_test_i)
    y_train_cv.append(y_train_i[0:train_num])
    y_valid_cv.append(y_train_i[train_num:])
    y_test_cv.append(y_test_i)
    
exp_num = 1    # the i-th experiment for 5-fold cross-validation

X_train = X_train_cv[exp_num]
X_valid = X_valid_cv[exp_num]
X_test = X_test_cv[exp_num]

def zero_one(y):
    y_ = np.zeros([y.shape[0],2])
    for i in range(y.shape[0]):
        if y[i] == 0:
            y_[i] = np.array([1,0])
        else:
            y_[i] = np.array([0,1])
    return y_

y_train = zero_one(y_train_cv[exp_num])
y_valid = zero_one(y_valid_cv[exp_num])
y_test = zero_one(y_test_cv[exp_num])
    
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')

# Some preprocessing
#features = preprocess_features(features)
#print(features.shape[1])

### in Haoyu's case, feature vector is not sparse ###

if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
# placeholder with [] shape takes a single scalar value directly. 
# placeholder with [None] shape takes a 1-dimensional array.
# placeholder with None shape can take in any value while computation takes place.
#https://stackoverflow.com/questions/46940857/what-is-the-difference-between-none-none-and-for-the-shape-of-a-placeh

placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape = (node_num, bert_dim)),
    'X': tf.placeholder(tf.int32, shape = (None, 2)),
    'y': tf.placeholder(tf.int32, shape = [None, 2]),
    #'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),    # y_train.shape[1]: 7
    #'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    #'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim = bert_dim, logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, X, y, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, X, y, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

def print_res(y_pred, y_true):
    y_pred = softmax(y_pred, axis=1)
    #print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    #print(y_pred)
    
    #print(precision_recall_fscore_support(y_true, y_pred, average='binary'))
    target_names = ['class 0', 'class 1']
    print(classification_report(y_true[:,1], y_pred, target_names=target_names))
    
# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, X_train, y_train, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs], feed_dict=feed_dict)
    print("train:\n")
    print_res(outs[3], y_train)
    
    
    # Validation
    cost, acc, output_val, duration = evaluate(features, support, X_valid, y_valid, placeholders)
    print("valid:\n")
    print_res(output_val, y_valid)
    cost_val.append(cost)

    # Print results
    #print("prediction results:\n")
    #print(outs[3])
    #print("\n y_train:\n")
    #print(y_train)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    #if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #    print("Early stopping...")
    #    break

print("Optimization Finished!")

# Testing
test_cost, test_acc, output_test, test_duration = evaluate(features, support, X_test, y_test, placeholders)
print_res(output_test, y_test)

print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

