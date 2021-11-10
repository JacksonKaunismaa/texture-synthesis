import numpy as np
import cv2
from PIL import Image

def show(im):
    im_pil = Image.fromarray(im.astype(np.uint8))
    im_pil.show()
imload = cv2.imread("./neckarfront.jpeg", cv2.IMREAD_COLOR)
imload = cv2.cvtColor(imload, cv2.COLOR_BGR2RGB)
imload = cv2.resize(imload, (224, 224))
x = np.random.uniform(low=0.0, high=255.0, size=(imload.shape))


def loss(im):
    return (np.square(im - imload)).sum()

def grad(im):
    return 2*(im - imload)

def line_search(grads, im):
    step = 1000
    alpha = 0.9
    best_step = 1000
    best_loss = float("inf")
    for _ in range(150):
        step_loss = loss(im - grads*step)
        if step_loss < best_loss:
            best_loss = step_loss
            best_step = step
        step *= alpha
    return best_step, best_loss
show(imload)
show(x)
for __ in range(10):
    gradient = grad(x)
    next_step, next_loss = line_search(gradient, x)
    print("loss now", loss(x))
    print("loss next", next_loss)
    print("next step", next_step)
    x -= gradient*next_step
    show(x)
