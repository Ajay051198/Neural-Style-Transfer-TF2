import image_utils
import NST_utils
import tensorflow as tf


c_path = 'spaceX.jpg'
s_path = 'winter.jpg'
style_image = image_utils.load_image(s_path)
content_image = image_utils.load_image(c_path)
# utils.display_image(img, 'starry_night')


train_steps = 120
image = NST_utils.train_for_steps(train_steps, style_image, content_image)
image_utils.display_image(image)
