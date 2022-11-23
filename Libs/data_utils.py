import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt



def plot_data_from_dicts_list(dicts_list):

  fig = plt.figure()
  ax = plt.subplot()


  # Over the possible recorder loss/metrics in disctionary form
  for dict_history in dicts_list:
    # The various type of metrics (per pixel, per channel, etc)
    for key, value in dict_history.items():
      # Plot the history
      ax.plot(value, label = f"{key}", linewidth=1.0 )

  # Add some info
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
            ncol = 4, fancybox=True, shadow=True)
  plt.show()


def load_train(filename, dim_ordering = 'tf', debug = True, bgr_to_rgb = False) :
    import joblib
    # Expects files to be loaded and pickled as RGB image 
    X = joblib.load(filename)['X']

    if debug :
        print("Color range pre: ", np.min(X), np.max(X) )


    # Scale color range from (0, 1) to (-1, 1)
    X = X * 2 - 1
    assert np.min(X) >= -1.0 and np.max(X) <= 1.0
    if debug :
        print("Color range: ", np.min(X), np.max(X) )
    
    if bgr_to_rgb :
        X = X[:, :, :, [2, 1, 0]]
    
    assert X.shape[-1] == 3
    if dim_ordering == 'th' :
        # Change dim_ordering from TF (count_images, height, width, 3) to theano (count_images, 3, height, width)
        X = np.swapaxes(X, 2, 3)
        X = np.swapaxes(X, 1, 2)
    elif dim_ordering == 'tf' :
        pass
    else :
        raise ValueError('Dim ordering %s is not supported' % dim_ordering)
    
    if debug :
        print('Shape:', X.shape)
    
    return X