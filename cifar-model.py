import tensorflow as tf
import numpy as np
from PIL import Image
import os
import re
import random
import pickle

FILTERS = [3, 64, 128, 256, 512]
SIZE = [3, 3, 3, 3, 3]
WEIGHTS = [1.0, 1.0, 1.0, 1.0]
START_SIZE = 32
TRAINING = False
EPOCHS = 10000
TRACKING = 100
LEARNING_RATE = 0.05
IMG_NAME = "./blue-stones.jpg"
LOG_PATH = "/tmp/texture_synth/model5"
S_PATH = "./models"
S_NAME = "model-{}"
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
BATCH_SIZE = 40
CLASS_NUM = 10

def convolve_getter(name, shape):
    weight = tf.get_variable(name, shape=shape, trainable=TRAINING)  # by default uses glorot_uniform initializer (pretty good)
    return weight

def convolve_once(input_tensor, convolver):
    # the most basic convolutional layer, including the convolution, batch norm, relu non-linearity
    conv = tf.nn.conv2d(input_tensor, convolver, strides=[1, 1, 1, 1], padding="SAME")
    conv_ = tf.layers.batch_normalization(conv, axis=-1, training=TRAINING, scale=False, center=True)
    return tf.nn.relu(conv_)


def pool_block(input_tensor, layer_name, idx):
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        with tf.name_scope("weights"):
            convolve_matrix = convolve_getter("convolve", [SIZE[idx], SIZE[idx], FILTERS[idx-1], FILTERS[idx]])
        with tf.name_scope("convolve_pool"):
            convolved = convolve_once(input_tensor, convolve_matrix)
            pooled = tf.nn.max_pool(convolved, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        return pooled

def read_in_image(name):
    raw_img = tf.image.decode_jpeg(tf.read_file(name), channels=FILTERS[0])
    shaped = tf.image.resize_images(raw_img, [START_SIZE, START_SIZE])
    return tf.expand_dims(shaped, 0)

def display_img(np_array):
    pil_img = Image.fromarray((np_array*1.).astype(np.uint8), "RGB")
    pil_img.show()


def attempt_classify(given_img):
    next_img = given_img
    for i in range(1, len(FILTERS)):
        next_img = tf.identity(pool_block(next_img, f"layer-{i}", i), name=f"outpt_layer-{i}")
    return next_img

def read_in_batch(batch_name):
    with open(batch_name, "rb") as f:
        data_load = pickle.load(f, encoding="bytes")
    lbls, img_data = data_load[b'labels'], data_load[b'data']
    img_arr = np.split(img_data, 3, axis=1)
    reshaped_imgs = [np.reshape(img_column, [10000, 32, 32]) for img_column in img_arr]
    proper_imgs = np.stack(reshaped_imgs, axis=3)
    return lbls, proper_imgs

def build_dataset(batch_names):
    all_data = list(map(read_in_batch, batch_names))
    all_imgs = [tup[1] for tup in all_data]
    all_labels = [tup[0] for tup in all_data]
    combined_imgs = np.concatenate(all_imgs, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    batched_imgs = np.split(combined_imgs, combined_imgs.shape[0]//BATCH_SIZE)
    batched_lbls = np.split(combined_labels, combined_imgs.shape[0]//BATCH_SIZE)
    return batched_imgs, batched_lbls


def read_in_names(name_to_class_loc):
    with open(name_to_class_loc, "rb") as f:
        dd = pickle.load(f, encoding="bytes")
    return dd[b'label_names']

global_step_tensor = tf.Variable(0, trainable=False)

next_images = tf.placeholder(tf.float32, [None, START_SIZE, START_SIZE, 3])
inpt_labels = tf.placeholder(tf.int32, [None])
next_labels = tf.cast(tf.one_hot(inpt_labels, CLASS_NUM), tf.float32)

class_oupt = attempt_classify(next_images)
fc = tf.Variable(tf.random.truncated_normal([FILTERS[-1] * ((START_SIZE // (2**(len(FILTERS)-1)))**2), CLASS_NUM]), trainable=True, name="fc")
final_result = tf.matmul(tf.reshape(class_oupt, [-1, FILTERS[-1] * ((START_SIZE // (2**(len(FILTERS)-1)))**2)], name="final_result"), fc)

closs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_labels, logits=final_result))
copt = tf.train.AdamOptimizer(LEARNING_RATE, name="copt").minimize(closs, global_step=global_step_tensor)

class_choices = tf.argmax(tf.nn.softmax(final_result), 1, name="class_choices")
correct_ones = tf.equal(class_choices, tf.argmax(next_labels, 1), name="correct_ones")
acc = tf.reduce_sum(tf.cast(correct_ones, tf.float32), name="accuracy")

#---------------------------------------------------------------------------------------------------------------------------------------------------------#
class_to_name = read_in_names("./cifar-10-batches-py/batches.meta")
train_imgs, train_labels = build_dataset([f"./cifar-10-batches-py/data_batch_{x_val}" for x_val in range(1,6)])
test_imgs, test_labels = build_dataset(["./cifar-10-batches-py/test_batch"])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    loader = tf.train.Saver()
    try:
        loader.restore(sess, tf.train.latest_checkpoint(S_PATH))
    except ValueError:
        print("No model found, initializing random model")
        grapher = tf.summary.FileWriter(LOG_PATH, sess.graph)
        grapher.add_summary(tf.Summary(), 0)
    print("Doing initial testing of model...")
    test_cost = 0
    total_correct = 0
    for iteration, (te_x, te_y) in enumerate(zip(test_imgs, test_labels)):
        test_batch_cost, test_batch_acc = sess.run([closs, acc], feed_dict={next_images: te_x, inpt_labels: te_y})
        test_cost += test_batch_cost
        total_correct += test_batch_acc
    print(f"Initial testing loss was {test_cost} (did {iteration} episodes)")
    print(f"Initial total correct testing was {total_correct} (did {iteration*BATCH_SIZE} samples)")
    for ep in range(EPOCHS):
        tr_loss = 0
        for iteration, (tr_x, tr_y) in enumerate(zip(train_imgs, train_labels)):
            _useless__, training_loss = sess.run([copt, closs], feed_dict={next_images: tr_x, inpt_labels: tr_y})
            tr_loss += training_loss
        print(f"Training loss was {tr_loss/iteration} (did {iteration} episodes)")
        test_cost = 0
        total_correct = 0
        for iteration, (te_x, te_y) in enumerate(zip(test_imgs, test_labels)):
            test_batch_cost, test_batch_acc = sess.run([closs, acc], feed_dict={next_images: te_x, inpt_labels: te_y})
            test_cost += test_batch_cost
            total_correct += test_batch_acc
        print(f"Testing loss was {test_cost/iteration} (did {iteration} episodes)")
        print(f"Total correct testing was {total_correct} (did {iteration*BATCH_SIZE} samples)")
        if ep % 5 == 0:
            saver.save(sess, os.path.join(S_PATH, S_NAME.format(tf.train.global_step(sess, global_step_tensor))))






