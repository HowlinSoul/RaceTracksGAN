from datetime import datetime
import os
import re   #regex
#import time
#import json
import tensorflow as tf
import numpy as np

#import logging
#logger = logging.getLogger(__name__)


# Comile The Model
def compileModel(model , lr = 1e-3 , loss = tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'] ):

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
  #sgd = tf.keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.5)
  model.compile(optimizer=optimizer, loss=loss, metrics = metrics)


# Callbacks
def getCallbacks(early_stop = True,early_stop_patience= 10, model_CheckPoint = True , exp_name = None , dateTime = True , save_best_only=False ,
                 save_freq = 'epoch', monitor = "val_loss", save_weights_only=True , mode = 'auto' , base_path ='/') :


  callbacks = []

   # If using Intersection over using, higher is better. Keras 'auto' mode can't infer this
  if re.search('IoU',monitor):
    mode = 'max'
  
  # Models folder
  exps_dir = os.path.join(base_path, 'model_experiments_')

  if not os.path.exists(exps_dir):
      os.makedirs(exps_dir)
  
  # If no name given , use a default name with date
  if exp_name is None :
    exp_name = 'Model Dated : '
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    exp_name = exp_name + '_' + str(now)
  
  #Add the date and time to the name of the experiment folder
  elif dateTime :
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    exp_name = exp_name + '_' + str(now)

  exp_dir = os.path.join(exps_dir, exp_name)
  
  if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)
    
  

  # ----------------
  # Model checkpoint
  # ----------------

  checkpoint_dir = None

  if model_CheckPoint:

    # Set parameters for weights checkpoints
    if save_weights_only :
      # Weights Checkpoint Folder
      checkpoint_dir = os.path.join(exp_dir, 'weight_ckpts')
      # File name
      checkpointName = 'cp_{epoch:02d,loss:.2f}.ckpt'
      # This is needed otherwise instead of overwriting the existing checkpoint, it creates a new one
       # Just a personal choice 
      if save_best_only :
        checkpointName = 'best_checkpoint.ckpt'        


    # Set parameters for the whole model checkpoints  
    else :  
      # Model Checkpoint Folder
      checkpoint_dir = os.path.join(exp_dir, 'model_ckpts')
      # !File name! -> For a model checkpoint, it is still a folder with assets inside
      checkpointName = 'model_{epoch:02d,loss:.2f}'
      #checkpointName = 'model_{epoch:02d}'
      # Just a personal choice to overwrite
      if save_best_only :
        checkpointName = 'best_model'

    # Path check
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

      # save_freq = 'epoch' -> save every epoch. save_freq = int -> save every int batches   
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, checkpointName), save_weights_only = save_weights_only,
                                                       save_best_only=save_best_only, monitor = monitor, save_freq = save_freq, mode=mode ) 
    # Finally add the callback
    callbacks.append(ckpt_callback)


  # ----------------
  # Visualize Learning on Tensorboard
  # ---------------------------------

  tb_dir = os.path.join(exp_dir, 'tb_logs')
  if not os.path.exists(tb_dir):
      os.makedirs(tb_dir)
    
  # By default shows losses and metrics for both training and validation
  tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                              profile_batch = 3,
                                              histogram_freq=1)  # if 1 shows weights histograms
  callbacks.append(tb_callback)


  # ----------------
  # Early Stopping
  # --------------

 
  
  # restore_best_weights=True
  if early_stop:
      es_callback = tf.keras.callbacks.EarlyStopping(monitor = monitor, patience=early_stop_patience , mode = mode)
      callbacks.append(es_callback)

  return [ callbacks , checkpoint_dir ]

  # ---------------------------------



# Define function to load weights from a checkpoint
def loadWeightsFromCheckpoint(model = None , pathToCheckpoints = None , experimentName = None, path_to_exp_dir = 'model_experiments_') :

    check_dir = None

    if pathToCheckpoints != None :
      check_dir = pathToCheckpoints  # Priority to a complete path
    elif experimentName != None :    
      check_dir = os.path.join(path_to_exp_dir , experimentName + '/weight_ckpts')
    else :
      print("No location provided")
      return
    
    if not os.path.exists(check_dir):
      print("Folder ' " + check_dir + " ' Not Found")
      return
    
    
    # Hope it gets the best model when checkpoints is set to 'save best only'. I need a general alternative for getting the best, not the latest
    latest = tf.train.latest_checkpoint(check_dir)

    if latest:
      print("Restoring from : ", latest)
      #model.load_weights(latest)
      model.load_weights(latest).expect_partial()    




# Define function to load and return the entire model
def loadModel(pathToModelCheckpoints = None , experimentName = None, best_model = True , path_to_exp_dir = 'model_experiments_') :


    check_dir = None
    checkpointNameFolder = ''

    # If the model was saved with 'Save best only'
    if best_model :
      checkpointNameFolder = '/best_model'

    # Entire path to checkpoint folder + model checkpoint name
    if pathToModelCheckpoints != None :
      check_dir = pathToModelCheckpoints + checkpointNameFolder  # Priority to a complete path

    elif experimentName != None : 
      checkpointNameFolder = '/best_model' # This is the main folder of the checkpoint file , {saved_model.pbtxt|saved_model.pb}. Works differently from load_weights for now
      check_dir = os.path.join(path_to_exp_dir , experimentName + '/model_ckpts' + checkpointNameFolder )
      
    else :
      print("No location provided")
      return
    
    if not os.path.exists(check_dir):
      print("Folder ' " + check_dir + " ' Not Found")
      return

    print("Restoring from : ", check_dir)
    
    #return tf.keras.models.load_model(check_dir)
    
    
    return tf.keras.models.load_model(check_dir)