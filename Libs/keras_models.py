import tensorflow as tf
#import numpy as np
#import logging
#logger = logging.getLogger(__name__)

# End to End Gan
def define_gan(g_model, d_model):
  # make the discriminator layer as non trainable
  d_model.trainable = False
  # get the noise and label input from the generator
  gen_noise, gen_label = g_model.input
  # get the output from the generator
  gen_output = g_model.output
  #connect image output and label input from generator as inputs to      #discriminator
  gan_output = d_model([gen_output,gen_label])
  #define gan model as taking noise and label and outputting a #classification
  model = tf.kreas.Model([gen_noise,gen_label],gan_output)
 
  return model


# Vanilla Generator Architecture
def make_generator_model(input_shape = 100, image_channels = 3):


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(6*6*512, use_bias=False, input_shape=(input_shape,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Reshape((6, 6, 512)))
    
    #8
    model.add(tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='valid', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    #16
    model.add(tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    #18
    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='valid', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    #20
    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='valid', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    #22
    model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='valid', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU() )

    #44
    model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU() )

    #46
    model.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='valid', use_bias=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU() )

   


    model.add(tf.keras.layers.Conv2DTranspose(image_channels, (3, 3), strides=(1, 1), padding='valid', use_bias=True, activation='tanh'))

    # Smooth the image
    #model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2) , strides = (1,1) , padding='valid' )  )



    #assert model.output_shape == (None, 48, 48, 1)

    return model
# Vanilla Discriminator Architecture
def make_discriminator_model(input_shape = [48, 48, 3]):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())


    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid ) )

    return model







## CGAN
# Generator
def CGAN_generator(latent_dim, n_class, channels = 1):
    label_input = tf.keras.layers.Input(shape=(1,))
    #Embedding layer
    em = tf.keras.layers.Embedding(n_class,20)(label_input) 
    em = tf.keras.layers.Dense(6*6*3)(em)
    em = tf.keras.layers.Reshape((6,6,3))(em)

    #image generator input
    image_input = tf.keras.layers.Input(shape=(latent_dim,))
    d1 = tf.keras.layers.Dense(6*6*509)(image_input)
    d1 = tf.keras.layers.LeakyReLU()(d1)
    d1 = tf.keras.layers.Reshape((6,6,509))(d1)
    
    # merge
    conv_part = tf.keras.layers.Concatenate()([d1,em])
    

    #12-14-28-30-32-64-128

    # Conv -> Batch -> Leaky [-> Dropout] # DCGAN Style
    for filters, stride, padding in zip([512,256,256,256,128,128,64,64] , [2,1,2,1,1,2,1] , ['same','valid','same','valid','valid','same','same']):
      
      conv_part = tf.keras.layers.Conv2DTranspose(filters, (3, 3), strides=(stride, stride), padding=padding, use_bias=False)(conv_part)
      conv_part = tf.keras.layers.BatchNormalization()(conv_part) 
      conv_part = tf.keras.layers.LeakyReLU()(conv_part)
      #conv_part = tf.keras.layers.Dropout(0.3)(conv_part)



    #output layer 
    out_layer = tf.keras.layers.Conv2D(channels,(3,3),activation='tanh', padding='same')(conv_part)
    #define model 
    model = tf.keras.Model([image_input,label_input],out_layer)
    return model


# Discriminator
def CGAN_discriminator(n_class, input_shape=(28,28,3), latent_space_output = False, use_classification = False):
    # label input
    in_labels = tf.keras.layers.Input(shape=(1,))
    # Embedding for categorical input
    em = tf.keras.layers.Embedding(n_class,20)(in_labels)

    # scale up the image dimension with linear activations
    d1 = tf.keras.layers.Dense(input_shape[0] * input_shape[1] * input_shape[2])(em)
    # reshape to additional channel
    d1 = tf.keras.layers.Reshape((input_shape[0],input_shape[1],input_shape[2] ))(d1)


    # image input
    image_input = tf.keras.layers.Input(shape=input_shape) 
    image_path = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False)(image_input)
    image_path = tf.keras.layers.BatchNormalization()(image_path) # DCGAN
    image_path = tf.keras.layers.LeakyReLU()(image_path)

    #  concate label as channel
    conv_part = tf.keras.layers.Concatenate()([image_path,d1])


    # Conv -> Batch -> Leaky -> [Dropout] x2
    for filters , stride in zip([128, 128, 256, 256] , [2, 2, 2, 2]):
      conv_part = tf.keras.layers.Conv2D(filters, (3, 3), strides=(stride, stride), padding="same", use_bias=False)(conv_part)
      conv_part = tf.keras.layers.BatchNormalization()(conv_part) # DCGAN
      conv_part = tf.keras.layers.LeakyReLU()(conv_part)
      #conv_part = tf.keras.layers.Dropout(0.3)(conv_part)



    flatten = tf.keras.layers.Flatten()(conv_part)
    
    # EXTRA
    dense = tf.keras.layers.Dense(64, use_bias=False)(flatten)
    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.LeakyReLU()(dense)

    # Discriminator output. Sigmoid activation function to classify "True" or "False"
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)


    if use_classification:
        # Conv -> Batch -> Leaky -> [Dropout] x2
        for filters , stride in zip([256, 512, 3] , [2, 2, 2, 1]):
          conv_part = tf.keras.layers.Conv2D(filters, (3, 3), strides=(stride, stride), padding="same", use_bias=False)(conv_part)
          conv_part = tf.keras.layers.BatchNormalization()(conv_part) # DCGAN
          conv_part = tf.keras.layers.LeakyReLU()(conv_part)
          #conv_part = tf.keras.layers.Dropout(0.3)(conv_part)
        
        GAP_2D = tf.keras.layers.GlobalAveragePooling2D(name= "GAP_2D")(conv_part)
        class_output = tf.keras.layers.Softmax(name="GAP_SOFTMAX_OUT")(GAP_2D)



    #define Input-Output model
    if use_classification and latent_space_output:
      model = tf.keras.Model([image_input,in_labels], [output, flatten, class_output] )
    elif latent_space_output:
        model = tf.keras.Model([image_input,in_labels],[output, flatten])
    elif use_classification:
      model = tf.keras.Model([image_input,in_labels],[output, class_output])
    else:
      model = tf.keras.Model([image_input,in_labels],output)

    
    
    return model




















## INFOGAN
# Generator
def infogan_generator(n_filters=128, noise_input=60, cat_inputs = 3, cont_inputs = 1, channels = 3):


    ## Noise input path
    noise_input = tf.keras.layers.Input(shape=(noise_input,), name = "noise_input")
    noise_input_path = tf.keras.layers.Dense(units=6*6*512, use_bias=False) (noise_input)
    noise_input_path = tf.keras.layers.BatchNormalization()(noise_input_path)
    noise_input_path = tf.keras.layers.LeakyReLU()(noise_input_path)
    noise_input_path = tf.keras.layers.Reshape((6,6,512), name="noise_reshape")(noise_input_path)

    ## Categorical input path
    cat_input = tf.keras.layers.Input(shape=(1,), name = "categorical_input")
    # Embedding layer
    cat_input_path = tf.keras.layers.Embedding(cat_inputs, 20)(cat_input)
    cat_input_path = tf.keras.layers.Dense( 6*6, activation= tf.keras.layers.LeakyReLU(0.2))(cat_input_path)
    cat_input_path = tf.keras.layers.Reshape((6,6, 1), name="cat_reshape")(cat_input_path)

    ## Continous input path
    cont_input = tf.keras.layers.Input(shape=(cont_inputs,), name = "continuous_input")
    cont_input_path = tf.keras.layers.Dense( 6*6, activation= tf.keras.layers.LeakyReLU(0.2))(cont_input)
    cont_input_path = tf.keras.layers.Reshape((6,6,1), name="cont_reshape")(cont_input_path)

       
    # Merge as filters
    conv_part = tf.keras.layers.Concatenate()([noise_input_path, cat_input_path, cont_input_path])

    
    # Conv -> Batch -> Leaky [-> Dropout] # DCGAN Style
    for filters, stride, padding in zip([512,256,128,128,64,32,32] , [2,1,2,1,1,2,1] , ['same','valid','same','valid','valid','same','same','same']):
      
      conv_part = tf.keras.layers.Conv2DTranspose(filters, (3, 3), strides=(stride, stride), padding=padding, use_bias=False)(conv_part)
      conv_part = tf.keras.layers.BatchNormalization()(conv_part) 
      conv_part = tf.keras.layers.LeakyReLU()(conv_part)
      #conv_part = tf.keras.layers.Dropout(0.3)(conv_part)


    # Final transposed convolutional layer, tanh activation
    output = tf.keras.layers.Conv2DTranspose(channels, kernel_size=(3, 3), strides=(1, 1), 
                                            padding="same", activation="tanh", use_bias=True)(conv_part)

    model = tf.keras.models.Model(inputs=[noise_input, cat_input, cont_input], outputs=output)
    
    return model

# Discriminator
def infogan_discriminator(n_class=10, cont_outputs = 1,  input_shape=(48, 48, 3), latent_space_output = False):
    # Build functional API model
    # Image Input
    image_input = tf.keras.layers.Input(shape=input_shape)
    # Start conv part
    conv_part = image_input


    # Conv -> Leaky -> Dropout x2
    for filters , stride in zip([64, 128, 128, 256, 256] , [2, 1, 2, 2, 1]):
      conv_part = tf.keras.layers.Conv2D(filters, (3, 3), strides=(stride, stride), padding="same", use_bias=False)(conv_part)
      conv_part = tf.keras.layers.BatchNormalization()(conv_part) # DCGAN
      conv_part = tf.keras.layers.LeakyReLU()(conv_part)
      #image_path = tf.keras.layers.Dropout(0.3)(image_path)
    

    # Flatten the convolutional layers
    flatten = tf.keras.layers.Flatten()(conv_part)

    # Latent Space
    latent_space = tf.keras.layers.Dense(64, use_bias=False, name = "latent_space_layer")(flatten)
    latent_space = tf.keras.layers.BatchNormalization()(latent_space)
    latent_space = tf.keras.layers.LeakyReLU()(latent_space)

    # Discriminator output. Sigmoid activation function to classify "True" or "False"
    d_output = tf.keras.layers.Dense(1, activation='sigmoid')(latent_space)


    # Auxiliary output. 
    q_dense = tf.keras.layers.Dense(32, use_bias=False)(latent_space)
    q_bn = tf.keras.layers.BatchNormalization()(q_dense)
    q_act = tf.keras.layers.LeakyReLU()(q_bn)

    # Classification (discrete output)
    clf_out = tf.keras.layers.Dense(n_class, activation="softmax", name = "Categoricals_Out_Layer")(q_act)

    # Gaussian distribution mean (continuous output)
    mu = tf.keras.layers.Dense(cont_outputs, name = "Means_Out_Layer")(q_act)

    # Gaussian distribution standard deviation (exponential activation to ensure the value is positive)
    sigma = tf.keras.layers.Dense(cont_outputs, activation=lambda x: tf.math.exp(x), name = "Variances_Out_Layer")(q_act)


    # Discriminator model 

    d_model = tf.keras.models.Model(inputs=image_input, outputs=d_output)

    # Auxiliary model 
    q_model = tf.keras.models.Model(inputs=image_input, outputs=[clf_out, mu, sigma])

    if latent_space_output:
      q_model = tf.keras.models.Model(inputs=image_input, outputs=[clf_out, mu, sigma, latent_space])

    

    return d_model, q_model





