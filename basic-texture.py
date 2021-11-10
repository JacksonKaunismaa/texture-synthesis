import tensorflow as tf
import numpy as np
from PIL import Image

EPOCHS = 10000
TRACKING = 100
LEARNING_RATE = 0.01
IMG_NAME = "./blue-stones.jpg"
TRAINING = False
START_SIZE = 64
FILTERS = [3, 64, 128, 256, 256]
SIZE = [3, 3, 3, 3, 3]
WEIGHTS = [1.0, 1.0, 1.0, 1.0]


def compute_gram_matrix(image_filters):
    # G[i, j] = sum_k F[i,k]*F[j,k], (b for batch, i,j are the x-y coords of the pixels we are multiplying, and c,d are the 2 filters)
    # this throws away spatial information and just finds correlations between the different vectors
    gram_mat = tf.linalg.einsum("bijc,bijd->bcd", image_filters, image_filters)
    inpt_size = tf.cast(tf.shape(image_filters)[1]*tf.shape(image_filters)[2], tf.float32)
    return gram_mat/inpt_size


def display_img(np_array):
    pil_img = Image.fromarray((np_array*1.).astype(np.uint8), "RGB")
    pil_img.show()


def get_texture_statistics(given_img, calculate_grams):
    grams = []
    next_img = given_img
    for i in range(1, len(FILTERS)):
        next_img, gram_matrix = pool_block(next_img, f"layer-{i}", i)
        grams.append(gram_matrix)
    return grams, next_img

def convolve_getter(name, shape):
    weight = tf.get_variable(name, shape=shape, trainable=TRAINING)  # by default uses glorot_uniform initializer (pretty good)
    return weight

def convolve_once(input_tensor, convolver):
    # the most basic convolutional layer, including the convolution, batch norm, relu non-linearity
    conv = tf.nn.conv2d(input_tensor, convolver, strides=[1, 1, 1, 1], padding="SAME")
#    conv_ = tf.layers.batch_normalization(conv, axis=-1, training=TRAINING, scale=False, center=True)
    return tf.nn.relu(conv)


def pool_block(input_tensor, layer_name, idx):
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        with tf.name_scope("weights"):
            conv1 = convolve_getter("conv1", [SIZE[idx], SIZE[idx], FILTERS[idx-1], FILTERS[idx]])
#            conv2 = convolve_getter("conv2", [SIZE[idx], SIZE[idx], FILTERS[idx], FILTERS[idx]])
        with tf.name_scope("convolve_pool"):
            convolved1 = convolve_once(input_tensor, conv1)
 #           convolved2 = convolve_once(convolved1, conv2)
            pooled = tf.nn.max_pool(convolved1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        return pooled, compute_gram_matrix(pooled)

def read_in_image(name):
    raw_img = tf.image.decode_jpeg(tf.read_file(name), channels=FILTERS[0])
    shaped = tf.image.resize_images(raw_img, [START_SIZE, START_SIZE])
    return tf.expand_dims(shaped, 0)

def display_img(np_array):
    pil_img = Image.fromarray((np_array*1.).astype(np.uint8), "RGB")
    pil_img.show()


textured_img = read_in_image(IMG_NAME)/255.
#mu = tf.reduce_mean(textured_img)
#sigma = tf.minimum(mu/2, (1-mu)/2)
#rand_img = tf.Variable(tf.truncated_normal([1, START_SIZE, START_SIZE, FILTERS[0]], mean=mu, stddev=sigma), trainable=True, name="rand_start_img")
rand_img = tf.Variable(tf.random_uniform([1, START_SIZE, START_SIZE, FILTERS[0]], minval=0.0, maxval=1.0), trainable=True, name="rand_start_img")
tstats, _toutput = get_texture_statistics(textured_img, True)
rstats, _routput = get_texture_statistics(rand_img, True)

tloss = tf.add_n([WEIGHTS[i] * tf.reduce_mean(tf.square(tstats[i]-rstats[i])) for i in range(len(FILTERS)-1)])
topt = tf.train.AdamOptimizer(LEARNING_RATE, name="opt", epsilon=1e-5).minimize(tloss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(EPOCHS):
        l, __ = sess.run([tloss, topt])
        if _ % TRACKING == 0:
            print(l)
            final = (sess.run(rand_img)*255).astype(np.uint8)
            img = Image.fromarray(final[0], "RGB")
            img.show()

