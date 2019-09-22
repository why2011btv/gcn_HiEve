import tensorflow as tf
import datetime

def masked_softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss"""
    class_weights = tf.constant([[46072.0/2/43621.0, 46072.0/2/2451.0]])
    weights = tf.reduce_sum(class_weights * tf.cast(labels, dtype=tf.float32), axis=1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    weighted_losses = unweighted_losses * weights
    loss = tf.reduce_mean(weighted_losses)
    
    return loss


def masked_accuracy(preds, labels):
    """Accuracy"""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    
    return tf.reduce_mean(accuracy_all)
