import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image

SIZE = 32
BATCH_SIZE = 32
FILTERS = [3, 32, 64, 128, 64, 32, 3]   # first and last of FILTERS is always 3 so dont even put it
FILTER_SIZE = [3, 3, 3, 3, 3, 3, 3]       # size of filters for each layer
LEARNING_RATE = 0.0005
EPOCHS = 1000
S_PATH = "./models"
LOG_PATH = "/tmp/texture-synth/model1"
S_NAME = "model-{}"
IM_DENOISE = "./100.jpg"

def read_in_image(name):
    raw_img = tf.image.decode_jpeg(tf.read_file(name), channels=3)
    shaped = tf.image.resize_images(raw_img, [SIZE, SIZE])
    noised = shaped + tf.random_normal([SIZE, SIZE, 3], mean=0.0, stddev=np.random.uniform(20.0, 30.0))
    return noised, shaped

def make_dataset(filenames):
    _dataset = tf.data.Dataset.from_tensor_slices(filenames)
    _dataset = _dataset.map(read_in_image)
    _dataset = _dataset.batch(BATCH_SIZE).repeat().shuffle(buffer_size=100)
    _iterator = _dataset.make_one_shot_iterator()
    return _iterator

def convolve_getter(name, shape):
    weight = tf.get_variable(name, shape=shape)  # by default uses glorot_uniform initializer (pretty good)
    return weight

def convolve_once(input_tensor, convolver):
    # the most basic convolutional layer, including the convolution, batch norm, relu non-linearity
    conv = tf.nn.conv2d(input_tensor, convolver, strides=[1, 1, 1, 1], padding="SAME")
   # conv_ = tf.layers.batch_normalization(conv, axis=-1, training=TRAINING, scale=False, center=True)
    return tf.nn.relu(conv)


def convolve_block(input_tensor, layer_name, idx):
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        with tf.name_scope("weights"):
            convolve_matrix = convolve_getter("convolve", [FILTER_SIZE[idx], FILTER_SIZE[idx], FILTERS[idx-1], FILTERS[idx]])
        with tf.name_scope("convolve_pool"):
            convolved = convolve_once(input_tensor, convolve_matrix)
        return convolved

def remove_noise(some_imgs):
    next_img = some_imgs
    for i in range(1, len(FILTERS)):
        next_img = tf.identity(convolve_block(next_img, f"layer-{i}", i), name=f"outpt_layer-{i}")
    return next_img

def prep_img(name):
    im_read = Image.open(name)
    #im_shaped = im_read.resize((SIZE, SIZE), Image.ANTIALIAS)
    im_shaped = im_read
    im_shaped.show()
    arr_im = np.array(im_shaped).astype(np.float32)
    return np.expand_dims(arr_im, axis=0)

def display_img(np_arr):
    im_show = Image.fromarray(np_arr.astype(np.uint8), "RGB")
    im_show.show()

all_names = os.listdir("./cifar-100-python")
random.shuffle(all_names)
train_names = [f"./cifar-100-python/{name}" for name in all_names if 't' not in name]
test_names = [f"./cifar-100-python/{name}" for name in all_names if 't' in name]
TR_SIZE = len(train_names)
TE_SIZE = len(test_names)
gst = tf.Variable(0, trainable=False)

tr_iter = make_dataset(train_names)
te_iter = make_dataset(test_names)

#next_tr = tf.transpose(tr_iter.get_next(), [1, 0, 2, 3, 4])
#next_te = tf.transpose(te_iter.get_next(), [1, 0, 2, 3, 4])
tr_input, tr_outpt = tr_iter.get_next()
te_input, te_outpt = te_iter.get_next()




noised = prep_img(IM_DENOISE) + np.random.normal( loc=0.0, scale=25.0, size=[SIZE, SIZE, 3]).astype(np.float32)
display_img(noised[0])
denoised = remove_noise(noised)
#tr_bs = tf.shape(next_tr)[0]
#tr_input = tf.squeeze(tf.slice(next_tr, [0, 0, 0, 0, 0], [1, tr_bs, SIZE, SIZE, 3]), axis=0)
#tr_outpt = tf.squeeze(tf.slice(next_tr, [0, 0, 0, 0, 0], [1, tr_bs, SIZE, SIZE, 3]), axis=0)

#te_input = tf.squeeze(tf.slice(next_te, [1, 0, 0, 0, 0], [1, -1, -1, -1, -1]), axis=0)
#te_outpt = tf.squeeze(tf.slice(next_te, [1, 0, 0, 0, 0], [1, -1, -1, -1, -1]), axis=0)

# would like to do next_tr[:, 0, :, :, :]
tr_removed = remove_noise(tr_input)
te_removed = remove_noise(te_input)


loss = tf.reduce_mean(tf.square(tr_removed - tr_outpt))
tloss = tf.reduce_mean(tf.square(te_removed - te_outpt))
opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=gst)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    loader = tf.train.Saver()
    result = sess.run(denoised)[0]
    print(np.mean(result))
    display_img(result*255)
#    try:
#        loader.restore(sess, tf.train.latest_checkpoint(S_PATH))
#    except ValueError:
#        print("No model found, initializing random model")
#        grapher = tf.summary.FileWriter(LOG_PATH, sess.graph)
#        grapher.add_summary(tf.Summary(), 0)
#    pct_incr = 0.1
#    pct_track = 0.0
#    num_iters = TR_SIZE//BATCH_SIZE
#    for ep in range(EPOCHS):
#        tr_loss = 0
#        pct_track = 0.0
#        print(f"Epoch {ep}/{EPOCHS}: ", end="", flush=True)
#        for iteration in range(num_iters):
#            _useless__, training_loss = sess.run([opt, loss])
#            tr_loss += training_loss
#            if iteration/float(num_iters) >= pct_track:
#                print("#", end="", flush=True)
#                pct_track += pct_incr
#        print(f"\nTraining loss was {tr_loss/iteration} (did {iteration} episodes)")
#        test_cost = 0
#        for iteration in range(TE_SIZE//BATCH_SIZE):
#            test_batch_cost = sess.run(tloss)
#            test_cost += test_batch_cost
#        print(f"Testing loss was {test_cost/iteration} (did {iteration} episodes)")
#        if ep % 5 == 0:
#            saver.save(sess, os.path.join(S_PATH, S_NAME.format(tf.train.global_step(sess, gst))))
#







