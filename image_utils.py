import tensorflow as tf
import matplotlib.pyplot as plt
import os

def load_image(path_to_image, max_dim = 720):
    # Loads the image and resizes it such that the larger dim = 512
    img = tf.io.read_file(path_to_image)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)


    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def display_n_save_image(image, label = None):
    plt.axis('off')
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if label:
        plt.title(label)
        plt.savefig(os.path.join('style_transferred/', f"{label}"))
    plt.show()
