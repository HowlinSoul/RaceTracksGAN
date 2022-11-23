import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



class gan_metric:
  def __init__(self, name, GAN_module) -> None:
    # Metric identifier
    self.name = name
    # To distinguish between discriminator and generator
    self.GAN_module = GAN_module
    

# A generic metric wrapper for a discriminator network in GANs
class gan_discriminator_metric(gan_metric):

  def __init__(self, metric, name, GAN_module, **kwargs) -> None:
    # Init parent
    super().__init__(name, GAN_module)

    # Create a new instance of the metric for each input flow
    self.real_input_metric = metric( **kwargs)
    self.fake_input_metric = metric( **kwargs)
    
    # Save history
    self.real_metric_data = []
    self.fake_metric_data = []
  

  def update_metric(self, output_on_real_images, output_on_fake_images):
    # For smooting instead of tf.ones_like(real_output) , just put 0.9
    _ = self.real_input_metric.update_state(tf.ones_like(output_on_real_images), output_on_real_images)
    _ = self.fake_input_metric.update_state(tf.zeros_like(output_on_fake_images), output_on_fake_images)
  
  # During Training reset the state every epoch/step
  def reset_state(self):
    # Store data before resetting
    self.real_metric_data.append( self.real_input_metric.result())
    self.fake_metric_data.append( self.fake_input_metric.result())

    # Reset
    self.real_input_metric.reset_state()
    self.fake_input_metric.reset_state()
  
  # Return the history so far
  def get_history(self):
    return self.real_metric_data, self.fake_metric_data


  def print_results(self ):
    # = self.real_input_metric.result().numpy()
    real_result = self.real_input_metric.result()
    fake_result = self.fake_input_metric.result()

    #print( f"Accuracy on Real : {disc_acc_on_real_image} \n Accuracy on Fake : {disc_acc_on_fake_image} " )
    tf.print( f"{self.name} on Real {real_result:.4} \n{self.name} on Fake {fake_result:.4}")







# A generic metric wrapper for a generator network in GANs
class gan_generator_metric(gan_metric):

  def __init__(self, metric, name, GAN_module, **kwargs) -> None:
    # Init parent
    super().__init__(name, GAN_module)

    # Init Correspondant Tensorflow Metric
    self.metric = metric( **kwargs)
    
    # Save history
    self.metric_data = []
    
  
  # Update metric state
  def update_metric(self, disc_output):
    # For smooting instead of tf.ones_like(real_output) , just put 0.9
    _ = self.metric.update_state(tf.ones_like(disc_output), disc_output)
  
  # During Training reset the state every epoch/step
  def reset_state(self):
    # Store data before resetting
    self.metric_data.append( self.metric.result())

    # Reset
    self.metric.reset_state()
  
  # Return the history so far
  def get_history(self):
    return self.metric_data


  def print_results(self ):
    # Curret active state
    result = self.metric.result()

    tf.print( f"{self.name} {result:.4}")


# A generic metric wrapper for a generator network in GANs
class single_input_metric(gan_metric):

  def __init__(self, metric, name, GAN_module, **kwargs) -> None:
    # Init parent
    super().__init__(name, GAN_module)

    # Init Correspondant Tensorflow Metric
    self.metric = metric( **kwargs)
    
    # Save history
    self.metric_data = []
    
  
  # Update metric state
  def update_metric(self, *input_):
    # For smooting instead of tf.ones_like(real_output) , just put 0.9
    _ = self.metric.update_state(*input_)
  
  # During Training reset the state every epoch/step
  def reset_state(self):
    # Store data before resetting
    self.metric_data.append( self.metric.result())

    # Reset
    self.metric.reset_state()
  
  # Return the history so far
  def get_history(self):
    return self.metric_data


  def print_results(self ):
    # Curret active state
    result = self.metric.result()

    tf.print( f"{self.name} {result:.4}")




# Plot the results of some metrics on the same plot
def plot_metrics(metrics):

  for metric in metrics:
    # A metric for the discriminator
    if metric.GAN_module == 'discriminator':
      real_m , fake_m = metric.get_history()

      plt.plot(real_m, label= f"{metric.name} R", linewidth=1.5 )
      plt.plot(fake_m, label = f"{metric.name} F", linewidth=1.5 )
    
    else:
    # A metric for the generator
    #if metric.GAN_module == 'generator':
      history = metric.get_history()
      plt.plot(history, label= f"{metric.name}", linewidth=1.5 )


  plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
             ncol = 4, fancybox=True, shadow=True)
  plt.show()
  