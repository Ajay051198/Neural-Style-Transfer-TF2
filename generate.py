import image_utils
import NST_utils
import tensorflow as tf
import os


CONTENT_DIR = 'content/'
STLYE_DIR = 'style/'


for file in os.listdir(CONTENT_DIR):
    c_path = os.path.join(CONTENT_DIR, file)
    for style in os.listdir(STLYE_DIR):
        s_path = os.path.join(STLYE_DIR, style)
        style_image = image_utils.load_image(s_path, max_dim = 1080)
        content_image = image_utils.load_image(c_path, max_dim = 1080)
        train_steps = 40
        image = NST_utils.train_for_steps(train_steps, style_image,
                    content_image, file, style)

        image_utils.display_n_save_image(image, label = f"{file.split('.')[0]}_{style.split('.')[0]}")
