import tensorflow as tf
import os

def vgg_layers(layers):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layers]
    model = tf.keras.models.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content':content_dict, 'style':style_dict}


def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

  opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

def style_content_loss(outputs,
                       style_targets,
                       content_targets,
                       num_style_layers,
                       num_content_layers,
                       style_weight,
                       content_weight):

    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

def train_step(image,
               style_targets,
               content_targets,
               extractor,
               num_style_layers,
               num_content_layers,
               opt,
               style_weight,
               content_weight):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs,
                              style_targets,
                              content_targets,
                              num_style_layers,
                              num_content_layers,
                              style_weight,
                              content_weight)


  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


def train_for_steps(n_steps,
                    style_image,
                    content_image,
                    file,
                    style,
                    content_layers = None,
                    style_layers = None,
                    opt = None,
                    style_weight = 1e-2,
                    content_weight = 1e4):

    if content_layers is None:
        content_layers = ['block5_conv2']

    if style_layers is None:
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

    if opt is None:
        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    num_style_layers = len(style_layers)
    num_content_layers = len(content_layers)


    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)

    for step in range(n_steps):
        os.system('cls')
        print('Generating')
        print(f"[INFO] Working with {file} as content image")
        print(f"[INFO] Applying style from {style}")
        completion_percentage = ((step+1)/(n_steps))
        progress = completion_percentage*100

        print('|',end='')
        print('>'*int(progress),end='')
        print('-'*int(100-progress), end='')
        print('|', end = ' ')
        print(f"[{round(completion_percentage*100, 2)}%]")

        train_step(image,
                   style_targets,
                   content_targets,
                   extractor,
                   num_style_layers,
                   num_content_layers,
                   opt,
                   style_weight = 1e-2,
                   content_weight = 1e4)

    print('Finished Generating!')
    return image
