from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import decode_predictions
from PIL import Image
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
import time
import pickle
from tensorflow.keras.models import Model
import tensorflow.keras as ks
from scipy.optimize import minimize
from scipy.optimize import Bounds

model = VGG19()
SIZE = 224
#LR = 500.0
SHOW = 1000
#DROP = 35
STYLE_NAME = "./lights.jpeg"
IM_NAME = "./krutik.jpg"
#HISTORY_SIZE = 20
#BETA_1 = 0.9
#BETA_2 = 0.999
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 700


def prep_img(name):
    im_read = Image.open(name)
    im_shaped = im_read.resize((SIZE, SIZE), Image.ANTIALIAS)
    im_shaped.show()
    arr_im = np.array(im_shaped)
    return np.expand_dims(arr_im, axis=0)

def gram_matrix(inpt_tensor):
    gram = tf.linalg.einsum("bijc,bijd->bcd", inpt_tensor, inpt_tensor)
    return gram/(2.0*tf.cast(tf.shape(inpt_tensor)[1]*tf.shape(inpt_tensor)[2]*tf.shape(inpt_tensor)[3], tf.float32))

def display_img(np_arr):
    im_shaped = np.reshape(np_arr, [SIZE, SIZE, 3])
    im_show = Image.fromarray(im_shaped.astype(np.uint8), "RGB")
    im_show.show()

def pickle_sv(np_arr, name):
    with open(name, "wb") as p:
        pickle.dump(np_arr, p)

def pred(an_im):
    return np.argmax(model.predict(an_im))


def img_dot(img1, img2):
    # returns the dot product of 2 images that have like more than one filter and with batches and shit (ie. 4D tensors -> 1D dot  product)
    return tf.linalg.einsum("bijk,bijk->b", img1, img2)[0]

def np_img_dot(img1, img2):
    # does the thing as img_dot() but for numpy images
    im1_reshaped = np.reshape(img1, (SIZE*SIZE*3))
    im2_reshaped = np.reshape(img2, (SIZE*SIZE*3))
    return im1_reshaped.dot(im2_reshaped)


def replace_intermediate_layer_in_keras(kmodel, layer_id, new_layer):
    layers = [l for l in kmodel.layers]

    x = layers[0].output  # model input
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)   # 
        else:
            x = layers[i](x)

    new_model = Model(inputs=layers[0].input, outputs=x)
    return new_model

def compute_search_dir(grad_inpt):
    global cnt
    q = grad_inpt    # all histories are in order of [0, 1, 2, 3,..., k-2, k-1] so doing history[-1] is like history[k-1] (but only go back to [k-HISTORY_SIZE, k-HISTORYS_SIZE+1, ... k-2, k-1])
    alpha = np.array([])   # alpha is supposed to be like alpha[k-1] = rho[k-1] * dot(xhist[k-1], q) so alpha will be ordered [k-1, k-2, k-3, ... k-HISTORY_SIZE] backwards to all the history stuff)
#    print("alpha", alpha.shape)
    for i in range(1, HISTORY_SIZE+1):
        try:  # alpha is a scalar array so its easy
            alpha = np.concatenate((alpha, np.expand_dims(rho_history[-i] * np_img_dot(xval_history[:,:,:,-i], q), axis=0)), axis=0)
            q = q - alpha[i-1]*grad_history[:,:,:,-i]    # alpha[i-1] is the value just computed which is what the algo calls for
        except IndexError:
            cnt += 1
            print("b1")
            break
    #print("hist, i", len(rho_hist), i)
    try:
        gamma_k = np_img_dot(xval_history[:,:,:,-1], grad_history[:,:,:,-1]) / (1e-10 + np_img_dot(grad_history[:,:,:,-1], grad_history[:,:,:,-1]))
    except IndexError:
        gamma_k = 1.0
    search_dir = gamma_k * q
    for i in range(1, HISTORY_SIZE+1):
        try:
            beta_current = rho_history[i-1] * np_img_dot(grad_history[:,:,:,i-1], search_dir)
            search_dir = search_dir + xval_history[:,:,:,i-1] * (alpha[-i] - beta_current)
        except IndexError:
            print("b2")
            cnt += 1
            break
    return search_dir


style_img = prep_img(STYLE_NAME)
content_img = prep_img(IM_NAME)
cnt = 0

#gradient_avg = 0.0
#velocity_avg = 0.0

model = replace_intermediate_layer_in_keras(model, 3, ks.layers.AveragePooling2D(pool_size=(2,2), padding="valid", strides=(2,2)))
model = replace_intermediate_layer_in_keras(model, 6, ks.layers.AveragePooling2D(pool_size=(2,2), padding="valid", strides=(2,2)))
model = replace_intermediate_layer_in_keras(model, 11, ks.layers.AveragePooling2D(pool_size=(2,2), padding="valid", strides=(2,2)))
model = replace_intermediate_layer_in_keras(model, 16, ks.layers.AveragePooling2D(pool_size=(2,2), padding="valid", strides=(2,2)))
model = replace_intermediate_layer_in_keras(model, 21, ks.layers.AveragePooling2D(pool_size=(2,2), padding="valid", strides=(2,2)))
rand_img = np.expand_dims(np.random.uniform(low=0.0, high=255.0, size=[SIZE,SIZE,3]), axis=0).astype(np.float64)

model_inpt = model.input              # CONCLUSION: NORMAL SCALE + PREPROCESS GIVES BEST RESULTS
#dog1 = prep_img("./dog1.jpg")
#dog2 = prep_img("./dog2.jpg")
#dog3 = prep_img("./dog3.jpg")
#car1 = prep_img("./car1.jpg")
#car2 = prep_img("./car2.jpg")
#car3 = prep_img("./car3.jpg")
#print("DOGS NORMAL SCALE")
#print(decode_predictions(model.predict(dog1))[0][0])
#print(decode_predictions(model.predict(dog2))[0][0])
#print(decode_predictions(model.predict(dog3))[0][0])
#print("CARS NORMAL SCALE")
#print(decode_predictions(model.predict(car1))[0][0])
#print(decode_predictions(model.predict(car2))[0][0])
#print(decode_predictions(model.predict(car3))[0][0])
#print("DOGS /255 SCALE")
#print(decode_predictions(model.predict(dog1/255.))[0][0])
#print(decode_predictions(model.predict(dog2/255.))[0][0])
#print(decode_predictions(model.predict(dog3/255.))[0][0])
#print("CARS /255. SCALE")
#print(decode_predictions(model.predict(car1/255.))[0][0])
#print(decode_predictions(model.predict(car2/255.))[0][0])
#print(decode_predictions(model.predict(car3/255.))[0][0])
#dog1 = preprocess_input(dog1)
#dog2 = preprocess_input(dog2)
#dog3 = preprocess_input(dog3)
#car1 = preprocess_input(car1)
#car2 = preprocess_input(car2)
#car3 = preprocess_input(car3)
#print("PREPROCCESSED DOGS NORMAL SCALE")
#print(decode_predictions(model.predict(dog1))[0][0])
#print(decode_predictions(model.predict(dog2))[0][0])
#print(decode_predictions(model.predict(dog3))[0][0])
#print("PREPROCCESSED CARS NORMAL SCALE")
#print(decode_predictions(model.predict(car1))[0][0])
#print(decode_predictions(model.predict(car2))[0][0])
#print(decode_predictions(model.predict(car3))[0][0])
#print("PREPROCCESSED DOGS /255 SCALE")
#print(decode_predictions(model.predict(dog1/255.))[0][0])
#print(decode_predictions(model.predict(dog2/255.))[0][0])
#print(decode_predictions(model.predict(dog3/255.))[0][0])
#print("PREPROCCESSED CARS /255. SCALE")
#print(decode_predictions(model.predict(car1/255.))[0][0])
#print(decode_predictions(model.predict(car2/255.))[0][0])
#print(decode_predictions(model.predict(car3/255.))[0][0])




#quit()
#layers = [model.layers[index].output for index in range(2, 22)]
style_layers = [model.layers[1].output,
            model.layers[4].output,
            model.layers[7].output,
         model.layers[12].output,
         model.layers[17].output]
grams = [gram_matrix(layer) for layer in style_layers]
g1 = grams[0]
g2 = grams[1]
g3 = grams[2]
g4 = grams[3]
g5 = grams[4]
#g6 = grams[5]
#g7 = grams[6]
#g8 = grams[7]
#g9 = grams[8]
#g10 = grams[9]
#g11 = grams[10]
#g12 = grams[11]
#g13 = grams[12]
#g14 = grams[13]
#g15 = grams[14]
#g16 = grams[15]
#g17 = grams[16]
#g18 = grams[17]
#g19 = grams[18]
#g20 = grams[19]

content_layers = [model.layers[13].output]
f1 = content_layers[0]

weights = np.array([1.0/len(style_layers)]*len(style_layers))
#eval_func = K.function([model_inpt], [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20])
style_eval_func = K.function([model_inpt], [g1, g2, g3, g4, g5])
content_eval_func = K.function([model_inpt], [f1])

style_targets = style_eval_func([style_img])
content_targets = content_eval_func([content_img])

style_loss = tf.add_n([weights[i] * tf.reduce_sum(tf.square(style_targets[i] - grams[i])) for i in range(len(style_targets))])
content_loss = tf.add_n([tf.reduce_sum(tf.square(content_targets[i] - content_layers[i])) for i in range(len(content_targets))])
loss = STYLE_WEIGHT * style_loss + CONTENT_WEIGHT * content_loss
grads = K.gradients(loss, model_inpt)[0]

#div_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
#opt_func = K.function([model_inpt], [loss, grads, g1])


#first_loss, first_grads, first_gram = opt_func([rand_img])   # first thing it does it take a steepest descent step
#prev_grad = first_grads
#prev_img = rand_img
#gradient_avg = BETA_1 * gradient_avg + (1 - BETA_1) * first_grads
#velocity_avg = BETA_2 * velocity_avg + (1 - BETA_2) * np.square(first_grads)
#print("first_grad", np.shape(first_grads))
#rand_img = np.clip(rand_img - first_grads*LR, a_min=0.0, a_max=255.0)  # update image based on search_dir = -(H^-1#)g
#second_loss, second_grads, second_gram = opt_func([rand_img])  # then it recomputes gradients so we can make the first iteration of histories
#print("second_grad", np.shape(second_grads))
#print("diff", np.shape(second_grads-prev_grad))
#grad_history = np.expand_dims((second_grads - prev_grad)[0], axis=3)
#xval_history = np.expand_dims((rand_img - prev_img)[0], axis=3)   # but for the recomputing it never actually takes a new step so prev_grads is still what it was at the initialization image
#rho_history = np.expand_dims(1. / np_img_dot(second_grads-prev_grad, rand_img-prev_img), 0)



#bfgs_search = K.function([model_inpt], [loss, grads])






#immediate_grads = K.gradients(loss, layers[0])[0][0]
#other_func = K.function([model_inpt], [loss, immediate_grads, layers[0], g1, partial_loss])
#lowest_loss = float("inf")
#since_last_drop = 0
#streak = 0
#grad_diffs = []
#img_diffs = []
loss_only = K.function([model_inpt], [tf.cast(loss, tf.float64)])
grad_only = K.function([model_inpt], [tf.cast(grads, tf.float64)])
def loss_getter_func(current_img):
    return loss_only([current_img])
def grad_getter_func(current_img):
    return grad_only([current_img])
def overall_func(current_img):
    shaped_img = current_img.reshape([1, SIZE, SIZE, 3])
    computed_loss, computed_grads = loss_getter_func(shaped_img)[0], grad_getter_func(shaped_img)[0]
    transformed_grads = computed_grads.reshape([224*224*3])
    return computed_loss, transformed_grads
external_counter = 0
def callback_func(current_img):
    global external_counter
    if external_counter % SHOW == 0:
        current_img = np.reshape(current_img, [224, 224, 3])
        display_img(current_img)
        print(loss_getter_func(np.expand_dims(current_img, 0)))
    external_counter += 1
#lower_bounds = np.ones_like(rand_img) * 0.0
#upper_bounds = np.ones_like(rand_img) * 255.0
rand_img_shaped = rand_img.reshape([224*224*3])
print(rand_img_shaped.shape)
#lower_bounds = np.ones_like(rand_img_shaped) * 0.0
#upper_bounds = np.ones_like(rand_img_shaped) * 255.0
result = minimize(overall_func, rand_img_shaped,
                  method="L-BFGS-B",
                  jac=True,
                  options={"maxiter": 10000},
                  bounds=Bounds(0, 255),
                callback=callback_func
                 )
pickle_sv(result, "final_img.p")
#for i in range(1, 100000):
##    loss_value, grads_value = bfgs_search([rand_img])
##    loss_value, grads_value, gram_1 = opt_func([rand_img])
##
##    gradient_avg = BETA_1 * gradient_avg + (1 - BETA_1) * grads_value
##    velocity_avg = BETA_2 * velocity_avg + (1 - BETA_2) * np.square(grads_value)
##    gradient_hat = gradient_avg / (1 - BETA_1**i)
##    velocity_hat = velocity_avg / (1 - BETA_2**i)
##    adam_update = gradient_hat / (np.sqrt(velocity_hat) + 1e-8)
##    #rmsprop_update = grads_divided
##    rand_img = np.clip(rand_img - LR * adam_update, a_min=0.0, a_max=255.0)
##    search_direction = compute_search_dir(grads_value)
##     xval_diff = rand_img - prev_img
##     grad_diff = prev_grad - grads_value
##     rho_value = 1. / np_img_dot(grad_diff, xval_diff)
##     hessian_guess = (identity - rho_value * xval_diff * grad_diff) * hessian_guess * (identity - rho_value * grad_diff * xval_diff) + rho_value* xval_diff * xval_diff
##     search_direction = hessian_guess * grads_value
##     prev_img = rand_img
##     prev_grads = grads_value
##    rand_img = np.clip(rand_img - grads_value*LR, a_min=1.0, a_max=255.0)  # update image based on search_dir = -(H^-1)g
##    try:
##        if grad_history.shape[3] >= HISTORY_SIZE:   # update histories
##            grad_history = grad_history[:,:,:,1:]
##        if xval_history.shape[3] >= HISTORY_SIZE:
##            xval_history = xval_history[:,:,:,1:]
##        if rho_history.shape[0] >= HISTORY_SIZE:
##            rho_history = rho_history[1:]
##    except IndexError:
##        print("shoud only see this once")
#    #print("before rho, xval, grad")
#    #print(rho_history.shape, xval_history.shape, grad_history.shape)
#    #print("new stuff rho, xval, grad")
#    #print((1. / np_img_dot(grads_value-prev_grad, rand_img-prev_img)).shape, np.expand_dims((rand_img-prev_img)[0], axis=3).shape, np.expand_dims((grads_value-prev_grad)[0], axis=3).shape)
# #   print(grad_history.shape, xval_history.shape, rho_history.shape)
# #   print([np.mean(grad_history[:,:,:,x]) for x in range(grad_history.shape[3])])
# #   print([np.mean(xval_history[:,:,:,x]) for x in range(xval_history.shape[3])])
# #   print(list(rho_history))
##    grad_history = np.concatenate((grad_history, np.expand_dims((grads_value - prev_grad)[0], axis=3)), axis=3)
##    xval_history = np.concatenate((xval_history, np.expand_dims((rand_img - prev_img)[0], axis=3)), axis=3)
##    rho_history = np.concatenate((rho_history, np.expand_dims((1. / (1e-10 + np_img_dot(grads_value-prev_grad, rand_img-prev_img))), axis=0)), axis=0)
##    #print("rho, xval, grad")
##    print(rho_history.shape, xval_history.shape, grad_history.shape)
#    #print(cnt)
#    #print(len(grad_history), len(xval_history), len(rho_history))
#    if i % SHOW == 0:
##    if i % SHOW in [996, 997, 998, 999]:
#        display_img(rand_img)
#        print("loss", loss_value)
#        print("grad mean", np.mean(grads_value))
##        print("gram mean", np.mean(gram1_so_far))
##        the_loss, hopefully_good, acts, gram1, ploss = other_func([rand_img])
##        pickle_sv(the_loss, f"immediate_loss({i%SHOW}).pickle")
##        pickle_sv(hopefully_good, f"immediate_grads({i%SHOW}).pickle")
##        pickle_sv(ploss, f"partial_loss({i%SHOW}.pickle")
##        pickle_sv(acts, f"act1({i%SHOW}).pickle")
##        pickle_sv(gram1, f"gram1({i%SHOW}).pickle")
##        pickle_sv(targets[0], f"targ1({i%SHOW}).pickle")
##        pickle_sv(rand_img, f"latest({i%SHOW}).npy")
##        pickle_sv(grads_value, f"grads({i%SHOW}).p")
##        if i % SHOW == 999:
##            quit()
#    #time.sleep(0.5)
#    if loss_value > lowest_loss:
#        streak += 1
#    else:
#        streak = 0
#    if streak >= DROP:
#        streak = 0
#        LR /= 2.
#        print(f"LR reduced by half after {since_last_drop}/{DROP} iterations ->", LR)
#        since_last_drop = 0
#    lowest_loss = min(loss_value, lowest_loss)
#    since_last_drop += 1
# #   prev_grad = grads_value
#    #prev_img = np.copy(rand_img)
