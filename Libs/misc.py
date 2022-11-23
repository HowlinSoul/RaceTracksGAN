import json
import os
import tensorflow as tf


# Load and return an object from a json file
def load_json_object(path_to_json_file):
  with open(path_to_json_file) as json_file:
      return json.load(json_file)

# Serialize object to JSON
def save_object_to_json(path_to_file, data):
  # Check if path exists, create
  os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
  # Write to file
  with open(path_to_file, 'w') as f:
        json.dump(data, f)



# Log the models architercutre to wandb
def wandb_log_models_architecture(models, wandb, path_to_file, models_name = ["model_plot"]):

    for model, model_name in zip(models, models_name):
        # Overwrite files, no need to keep them on colab
        path_to_model_plot = path_to_file + "model_plot.jpeg"
        _ = tf.keras.utils.plot_model(model, show_shapes=True, to_file= path_to_model_plot)
        model_plot = tf.io.decode_png( tf.io.read_file(path_to_model_plot) )

        wandb_image = wandb.Image(model_plot, caption = model_name)    
        wandb.log({"Model Architecture": wandb_image})


'''
Custom dict log
wandb.log({"loss": 0.314, "epoch": 5,
           "inputs": wandb.Image(inputs),
           "logits": wandb.Histogram(ouputs),
           "captions": wandb.Html(captions)})
'''



# Decaying Learning Rate. Used when the optimizer doesen't call the learning rate function with step parameter
class DecayLRWrapper():
    def __init__(self, init_lr, min_lr, decay_rate, decay_steps):
        self.step = 1
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps


    def decayed_learning_rate_mtresh_ffile(self):
        self.step+=1
        return max(self.init_lr * self.decay_rate ** (self.step / self.decay_steps), self.min_lr)





# Exponential Decay Learning Rate with minimum Threshold
def decayed_learning_rate_mtresh(step):
  initial_learning_rate= 1e-2
  decay_rate= 0.96
  decay_steps= 550 # Around epoch 85 it goes to minimum (with current settings)
  min_learning_rate= 2e-4

  return max(initial_learning_rate * decay_rate ** (step / decay_steps), min_learning_rate)
