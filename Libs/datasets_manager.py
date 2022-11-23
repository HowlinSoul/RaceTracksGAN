import numpy as np
import keras
import tensorflow as tf
import image_utils
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 1234


#0-9 digits dataset
def mnist_digit_dataset():
  from keras.datasets import mnist


  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  X_ = np.concatenate((x_train, x_test), axis=0)
  Y_ = np.concatenate((y_train, y_test), axis=0)

  X_ = X_.reshape(X_.shape[0], 28, 28, 1).astype('float32')

  
  print( f"Input Shape : {X_.shape} \nOutput Shape : {Y_.shape}" )
  

  return X_, Y_


def dummy_RGB_dataset():
  image = np.zeros( shape = [28,28,3])

  X_ = np.zeros( shape = [90,28,28,3] )
  Y_ = np.zeros( shape = [90,1] )


  # RED IMAGES
  X_[:30,2:14,2:14,0] = tf.random.normal( X_[:30,2:14,2:14,0].shape, 160, 10, tf.float32)
  Y_[:30] = 0
  # GREEN
  X_[30:60,0:14,14:28,1] = tf.random.normal( X_[30:60,0:14,14:28,1].shape, 160, 10, tf.float32)
  Y_[30:60] = 1
  # BLUE
  X_[60:90,10:20,10:20,2] = tf.random.normal( X_[60:90,10:20,10:20,2].shape, 160, 10, tf.float32)
  Y_[60:90] = 2

  print( f"Input Shape : {X_.shape} \nOutput Shape : {Y_.shape}" )
  
  return X_, Y_

def dummy_RGB_dataset_transformed():

  # Trasform data
  trasformations = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomTranslation(height_factor = 0.05, width_factor=0.05),
  ])
  

  X_, Y_ = dummy_RGB_dataset()

  X_ = trasformations(X_)

  return X_, Y_

# Pixel Art Characters/Objects dataset
def pixelart_dataset(batch_size = 40):
  base_path = '/content/drive/My Drive/'
  save_path = base_path + 'DATA_Rep/pixelart-48x48.npy'
  # The dataset is stored normalized (-1,1)
  X = np.load(save_path)[:-12000] # Memory issues
  print("Data shape : ", X.shape)

  tf_dataset = to_tensorflow_dataset(X, batch_size )
  dists = extract_distribution(image_utils.denormalize_images(X) )

  # Plot/display some infos
  for x in tf_dataset:
    image_utils.display_multiple_image(x[:8] , size=[1500,2000] , denormalize=True)
    break

  print(tf_dataset)
  print(dists)

  return tf_dataset, dists



def to_tensorflow_dataset(data, batch_size ):

  BUFFER_SIZE = 20000  

  tf_dataset = tf.data.Dataset.from_tensor_slices( data )
  # Normalize
  #training_dataset = training_dataset.map(image_utils.normalize_image)
  # To One Hot
  #training_dataset = training_dataset.map(train_asses.to_one_hot_encode)
  # Batch and shuffle the data
  tf_dataset = tf_dataset.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True) # In Order to have the same batch Dimension, without a repeat dataset - To change  


  #valid_dataset = tf.data.Dataset.from_tensor_slices((X_[-BATCH_SIZE:],Y_[-BATCH_SIZE:]))
  # Normalize
  #valid_dataset = valid_dataset.map(image_utils.normalize_image)
  # To One Hot
  #valid_dataset = valid_dataset.map(train_asses.to_one_hot_encode)
  # To Batch
  #valid_dataset = valid_dataset.batch(BATCH_SIZE)

  return tf_dataset


def extract_distribution(data):
  px_dist = image_utils.extract_distribution(data , of_type = "pixel") 
  channel_dist = image_utils.extract_distribution(data , of_type = "channel")
  patch_dist = image_utils.extract_distribution(data , of_type = "patch", patch_shape= [2,2], patch_type = 'channel'  )

  distributions = {"Distribution per Pixel" : px_dist , "Distribution Per Channel" : channel_dist , "Distribution Per (2,2) Patch" :  patch_dist}
  return distributions


# TORCS TRACKS Dataset Generator And Data Augmentation
def TORCS_tracks_loader(batch_size = 40, preprocessing_function = None, class_mode = "sparse", apply_data_augmentation = False,
                         image_size = [28,28], dataset_dir = None, validation_split = 0, cval = 0 ):

    
    # Create training ImageDataGenerator object
    if apply_data_augmentation:
        train_data_gen = ImageDataGenerator(rotation_range=300,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='constant',
                                            zoom_range=[1.5,1.5],
                                            cval= cval,
                                            validation_split= validation_split,
                                            preprocessing_function=preprocessing_function)
    else:
        #train_data_gen = ImageDataGenerator(rescale=1./255 , validation_split= validation_split)
        train_data_gen = ImageDataGenerator(validation_split= validation_split, preprocessing_function=preprocessing_function)


    # Create generators to read images from dataset directory
    # -------------------------------------------------------

    # img shape
    img_h = image_size[0]
    img_w = image_size[0]

  
   
    train_gen = train_data_gen.flow_from_directory(dataset_dir,
                                                  batch_size=batch_size, 
                                                  class_mode= class_mode, # 'categorical' targets are directly converted into one-hot vectors, 'sparse' integers
                                                  shuffle=True,
                                                  color_mode = 'grayscale',  #default is rgb.  One of "grayscale", "rgb", "rgba"
                                                  subset="training", #extract training subset
                                                  target_size=(img_w, img_h),
                                                  interpolation='bilinear',
                                                  seed=SEED)
    

    valid_gen = train_data_gen.flow_from_directory(dataset_dir,
                                                  batch_size=batch_size, 
                                                  class_mode= class_mode, # 'categorical' targets are directly converted into one-hot vectors, 'sparse' integers
                                                  shuffle=True,
                                                  color_mode = 'grayscale',  #default is rgb.  One of "grayscale", "rgb", "rgba"
                                                  subset="validation", #extract training subset
                                                  target_size=(img_w, img_h),
                                                  interpolation='bilinear',
                                                  seed=SEED)

    # Display some sample
    x, _  = train_gen.next()
    print(f"Min : {np.min(x)}, Max : {np.max(x)} Values")
    image_utils.display_multiple_image(x[:8], denormalize=True, size=[1000,1000])

    # Inline return condition is rigged
    if validation_split == 0 :
      return train_gen 
    else:
      return train_gen, valid_gen




# Merge multiple data generators into a single dataset
class Merge_Generator(keras.utils.Sequence):
  def __init__(self, *generators):
    self.generators = generators

  #Denotes the number of batches per epoch
  def __len__(self): 
        #return int(np.floor(len(self.list_IDs) / self.batch_size))
        return 1000
  
  #Generate one batch of data
  def __getitem__(self, index):
        
        x_ = None
        y_ = None

        for generator in self.generators:

          if x_ is not None:
            # Next Samples
            nx_, ny_ = generator.next()
            # Concatenate in one batch
            x_, y_ = tf.concat([x_, nx_], axis=0), tf.concat([y_, ny_], axis=0)

          # First Iteration Get Samples
          x_, y_ = generator.next()
        return x_, y_

  # Shuffle data or stuff
  def on_epoch_end(self):
    for generator in self.generators:
        generator.on_epoch_end()


# Wrap a generator model to generate syntethic samples
class Syntethic_Model_Generator():
  def __init__(self, model, num_classes, noise_dim, batch_size = 16):

    self.generator = model
    self.num_classes = num_classes
    self.noise_dim = noise_dim
    self.batch_size = batch_size

  
  # Noise for the generator
  def sample_noise(self, batch_size):
      return tf.random.normal(shape=[batch_size, self.noise_dim])
  # Sample Random Labels
  def sample_labels(self, batch_size):
      return tf.random.uniform(maxval = self.num_classes , shape=[batch_size], dtype = tf.int32)

  # Used to generate Syntetich Data  (->Data Generator Keras)
  def next(self):
      # Tracks generates, not garbage images
      tracks_to_generate = self.batch_size
      tracks_generated = []

      while( len(tracks_generated) < tracks_to_generate ):
        
        noise_input = self.sample_noise(16)
        labels = self.sample_labels(16)

        # Generate images
        gen_images = self.generator.predict([noise_input, labels], verbose=0)
        # Check Images Validity
        valid_images_arr = image_utils.are_images_closed(gen_images, silent_mode = True)

        tracks_generated = tracks_generated + valid_images_arr
        # Discard the tracks in excess
        tracks_generated = tracks_generated[:tracks_to_generate]



      return np.array(tracks_generated), self.sample_labels(self.batch_size)









'''
    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                                  output_types=(tf.float32),
                                                  output_shapes=([None, img_h, img_w, 1]))                                              

'''
'''
    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                                  output_types=(tf.float32, tf.float32),
                                                  output_shapes=([None, img_h, img_w, 1], [None,]))



    train_dataset = tf.data.Dataset.from_generator(train_gen,
                                                  output_signature=(
                                                      tf.TensorSpec(shape=([None, img_h, img_w, 1]), dtype=tf.float32),
                                                      tf.TensorSpec(shape=([None,]), dtype=tf.float32) ))
'''
  
