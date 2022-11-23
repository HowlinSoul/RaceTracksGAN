import numpy as np
import tensorflow as tf
# Tensorflow Dist
import tensorflow_probability as tfp

import os

# Libs for Images
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import glob #it does pattern matching and expansion <- Retrieving filenames on system and such

# Custom data structures
import structures

#import logging
#logger = logging.getLogger(__name__)

# Image processing tools
import cv2

# Avoids 0 probabilities
TINY = 1e-8



# Plots several graphs based on the input distributions
def plot_channels_dist(dists_array, image_channels, title = None):
 
  # Channel colors
  channels = ['red','green','blue']

  # Grayscale image
  if image_channels == 1:
    channels = [ ['orange'] ]

  
  # Total Number of distributions to plot
  n_dists = len(dists_array)
  # Expected Columns
  k_columns = n_dists * image_channels
  
  #print(" K columns :" , k_columns)

  # Size of the plot based on how many figures
  plt.subplots(figsize=( 7+ 3*n_dists + 3*image_channels, 12+ n_dists ) )
  # Vertical margin
  plt.subplots_adjust( hspace = 0.15)

  # Add title
  if title is None:
    title = [ "Distribution " + str(k+1) for k in range(n_dists) ]

  # For each k dist, plot the relevant graphs
  for k, dist in enumerate(dists_array):

    dist_samples = dist.sample(10)
    # Extract the mean value per pixel per channel
    avg_mean_per_channel = np.mean(dist_samples, axis = 0 ) # used for the histogram

    for ch in range( image_channels ):
      
      

      # Column index
      column_j = ch + k*image_channels + 1

      

      # Row index
      row_1 = 0*n_dists*image_channels
      row_2 = 1*n_dists*image_channels
      row_3 = 2*n_dists*image_channels

     

      #Scatter plot
      #plt.subplot(3 ,  k_columns , row_1 + column_j)

      if ch == 1:
        plt.title(title[k])
      #plt.scatter(dist_samples[:, :,:,ch],  dist.prob(dist_samples)[:, :,:,ch] , color=channels[ch], alpha=0.4)

      # Histogram
      plt.subplot(3 ,  k_columns , row_2 + column_j )
      plt.hist( avg_mean_per_channel[:,:,ch])
      plt.axis('on')

      
      # Image Form
      plt.subplot(3 ,  k_columns , row_3 + column_j)
      plt.imshow( avg_mean_per_channel[:,:,ch], interpolation='nearest', aspect='auto' )
      plt.axis('off')

      # This has to be better
      '''
      f, axarr = plt.subplots(2, 2)
      axarr[0,0].imshow(img, cmap = cm.Greys_r)
      axarr[0,0].set_title("Rank = 512")
      axarr[0,0].axis('off')
      '''
      #print(row_1 + column_j ,"  " ,  row_2 + column_j , "   " , row_3 + column_j)


  plt.show()  

  return




# Extract the Mean and Standard deviation per pixels (If the images have just one channel, this will coincide with the per channel one)
# result -> [w,h]
def get_mean_std_per_pixel(image_dataset):
  
  assert image_dataset.shape[-1] == 3 or image_dataset.shape[-1] == 1

  mean_across_images = np.mean(image_dataset,axis=0)
  #print("Mean across images. New shape : \n",mean_across_images.shape)

  mean_across_channels = np.mean(mean_across_images,axis =-1) #mean per pixel
  std_across_channels = np.std(mean_across_images,axis =-1)  #std per pixel

  #print("Mean across channels. New shape : \n",mean_across_channels.shape)
  #print("STD across channels. New shape : \n",std_across_channels.shape)


  return mean_across_channels, std_across_channels

# Extract the Mean and Standard deviation per channel 
# result -> [w,h,ch]
def get_mean_std_per_channel(image_dataset):
  
  assert image_dataset.shape[-1] == 3 or image_dataset.shape[-1] == 1

  mean_per_channel = np.mean(image_dataset,axis = 0)
  std_per_channel = np.std(image_dataset,axis = 0)

  #print("Mean per channel . New shape : \n",mean_per_channel.shape)
  #print("Std per channel . New shape : \n",std_per_channel.shape)

  return mean_per_channel,std_per_channel


# Compute the average of a small window (patcg) on the images
# result -> [new_w,new_h,ch] or [new_w,new_h] depending on the mode (per pixel/per channel)
def get_mean_std_per_patch(image_dataset , patch_shape  = (2,2) , patch_type = 'channel' , overlapping = False):
    
    strides = patch_shape
    # Patches overlaps on same pixels
    if overlapping :
      strides = (1,1)
    
    
    avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size = patch_shape, strides= strides , padding='valid')

    # Channel Wise Patch
    if patch_type == 'channel':
      mean, std = get_mean_std_per_channel(image_dataset)
    # Pixel Wise Patch
    else:
      mean, std = get_mean_std_per_pixel(image_dataset)
      mean = tf.reshape( mean , shape =[mean.shape[0] , mean.shape[1] , 1])
      std = tf.reshape( std , shape =[std.shape[0] , std.shape[1] , 1])
    
    mean = avg_pool_2d( tf.expand_dims(mean , axis = 0) )
    std = avg_pool_2d( tf.expand_dims(std , axis = 0) )

    # Eliminate the batch size
    return mean[0], std[0]


# Main Hub for getting image pixels distributions
def extract_distribution(data, of_type = "channel", **kwargs ):

  # Channel Wise Patch
  if of_type == 'channel':
    mean, std = get_mean_std_per_channel(data)
  # Pixel Wise Patch
  elif of_type == 'patch':
    mean, std = get_mean_std_per_patch(data, **kwargs)
  elif of_type == 'pixel':
    mean, std = get_mean_std_per_pixel(data)
  else: raise Exception("Type Not Supported. Supported types 'pixel, channel, patch") 

  dist= tfp.distributions.Normal(loc=mean+ TINY, scale=std+ TINY)

  return dist





# Show multiple images on screen
def display_multiple_image(images_array , inline = True ,  size = [300,300] , save_param = {"Save Image" : False,"Path" : None, "filename" : "img.jpg"} , denormalize = False):
  
  
  # N images
  n_samples = len(images_array)
  # Shape of each image
  image_shape = images_array[0].shape
  
  # Shape of matplot subplots
  shape = [1,n_samples]
  if not inline:
    shape = ( int( np.sqrt(n_samples)) , int( np.sqrt(n_samples) ) )


  #fig = plt.figure(figsize=figsize)
  px = 1/plt.rcParams['figure.dpi']  # inches to pixel conversion
  #plt.subplots(figsize=( image_shape[0]*modify_size_by*px, image_shape[1]*modify_size_by*px) )
  _ = plt.figure(figsize=( size[0]*px, size[1]*px))
  # Eliminate all the padding/margin -> 1px of
  plt.subplots_adjust(wspace=1*px, hspace=1*px)

  for i in range( n_samples ):
      plt.subplot(shape[0], shape[1] , i+1)

      image = images_array[i]

      if denormalize:
         image = denormalize_image(image)

      # The standard one
      cmap = 'viridis'
      # In order to plot [w,h,1] images
      if image.shape[2] == 1:
        image = image[:, :, 0]
        cmap = 'gray'

      plt.imshow(image, cmap = cmap)
      plt.axis('off')
  
  if save_param["Save Image"]:
    # Check or Create Path
    if not os.path.exists(save_param["Path"]):
      os.makedirs(save_param["Path"])
    # Save Image to File
    plt.savefig(save_param["Path"] + save_param["filename"])
      
  
  plt.show()



# Display image on cell
def display_image(image_array, size = [100,100]):
  px = 1/plt.rcParams['figure.dpi']  # inches to pixel conversion

  # Modify figure size
  _ = plt.figure(figsize=( size[0]*px, size[1]*px))
  plt.axis('off')

  # Tensorflow related assignment
  image = image_array

  # The standard one
  cmap = 'viridis'
  # In order to plot [w,h,1] images
  if image_array.shape[2] == 1:
    image = image[:, :, 0]
    cmap = 'gray'

  plt.imshow(image_array,cmap = cmap)
  plt.show()



# Stack multiple images togheter by row/column
def stack_multiple_images(images_array , inline = False):
  n_images = len(images_array)

  # Square size of images
  size = int(np.sqrt(n_images))

  image_vertical_slices = []
  # First stack vertically
  for i in range(0, n_images ,size):
    new_image = np.concatenate((images_array[i:i+size]) , axis =0)
    image_vertical_slices.append(new_image)

  stacked_image = None
  # Stack Vertical slices horizontally
  for image_slice in image_vertical_slices:
    if stacked_image is None:
      stacked_image = image_slice
    else:
      stacked_image = np.concatenate((stacked_image , image_slice) , axis = 1)

  return stacked_image


# Save image to File Using PIL
def save_image(image_array , path_to_file):
  image = Image.fromarray(np.uint8( image_array))
  image.save(path_to_file)

# Generate A GIF image using image.io
def generate_GIF(image_source_path, dest_file):

  with imageio.get_writer(dest_file, mode='I') as writer:
    
    filenames = glob.glob(image_source_path + 'image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)



#----- DATASETS MANIPULATION/GENERATIONS ----- #
# INPUT: IMAGE DATASET ORGANIZED IN FOLDER FOR EACH CLASS
def mod_images_to_file(original_data_path, new_data_path = None, operations = ["BITNOT"]):

    for class_dir in os.listdir(original_data_path):
      #print(class_dir)
      
      # Mac files
      if class_dir == ".DS_Store":
        continue

      for images in os.listdir( os.path.join(original_data_path, class_dir)):
        
        # check if the image ends with png
        if (images.endswith(".png")):
            #image = cv2.imread(os.path.join(original_data_path, class_dir, images), cv2.IMREAD_GRAYSCALE)
            image = np.array(Image.open(os.path.join(original_data_path, class_dir, images)).convert('L'))
            #image.shape

            img = np.array(image, dtype = np.uint8)
            for op in operations:
                if op == "BITNOT":
                    img = cv2.bitwise_not(img)
                elif op == "FLOODFILL":
                    # erode
                    img = cv2.bitwise_not(img)
                    #mask = np.zeros((np.array(img.shape)+2), np.uint8)

                    #print("FLOOD FILL")
                    _ = cv2.floodFill( img, None, (0,0), (255))
                else:
                    raise Exception(f"Operation not implemented : {op}")
              

            #image_r = np.array(image)
            #image_r = image_utils.normalize_images(image_r)
            #image_r[ image_r > 0 ] = 1
            #image_r[ image_r <= 0 ] = -1
            #img = np.array(image_r, dtype = np.uint8)
            
            #img = np.array(image, dtype = np.uint8)
            
            #img = image_utils.normalize_images(img)
            #img = np.expand_dims(image, -1)
            #print(f"{img.shape} {np.max(img)} {np.min(img)}")
            
            #print(f"NORMAL image {images}")
            #display_multiple_image( [np.expand_dims(image, -1), np.expand_dims(img, -1)])

            #print("BITNOT")
            #mask = cv2.bitwise_not(img)
            #image_utils.display_multiple_image( [np.expand_dims(mask, -1)])

             # SAVE IMAGE TO FILE
            if new_data_path:
                # Check Folders Existence
                if not os.path.exists(os.path.join(new_data_path, class_dir)):
                    os.makedirs(os.path.join(new_data_path, class_dir))

                cv2.imwrite(os.path.join(new_data_path, class_dir, images), img)

            

           
            
       
          




#----- IMAGE PROCESSING ----- #

def normalize_image(x_,y_): 
  
  #x_ = tf.cast(x_, tf.float32)
  x_ = (x_-127.5)/127.5

  return x_,  y_

def normalize_images(images_arr):
  # To avoid tensorflow eager errors
  images_arr = np.array(images_arr)
  return (images_arr-127.5)/127.5

def denormalize_images(images_arr):
  return tf.cast( (images_arr*127.5 + 127.5) , tf.uint8)
  
  #return (images_arr*127.5 + 127.5).astype(int)
  

def denormalize_image(image):
  image = np.array((image*127.5 + 127.5) ).astype(int)

  return tf.cast( image , tf.uint8)






# Preprocess into skeleton method and then check
#Image in grayscale needed
# CUSTOM METHOD
def extract_skeleton(img):
  # Binary image
  img[ img > 0 ] = 255
  img[ img <= 0 ] = 0

  img = np.reshape(img, (img.shape[0],img.shape[1]))

  #img.dtype
  # Step 1: Create an empty skeleton
  size = np.size(img)
  skel = np.zeros(img.shape, np.float32) #same type as the original image

  # Get a Cross Shaped Kernel
  element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

  # Repeat steps 2-4
  while True:
      #Step 2: Open the image - Opening is simply Erosion followed by Dilation.
      open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
      #Step 3: Substract open from the original image
      temp = cv2.subtract(img, open)

      #Step 4: Erode the original image and refine the skeleton
      eroded = cv2.erode(img, element)

      skel = cv2.bitwise_or(skel, temp)
      img = eroded.copy()
      # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
      if cv2.countNonZero(img)==0:
          break
  
  return np.reshape(skel, (img.shape[0], img.shape[1], 1) )


# CV2 METHOD
def extract_skeleton_inbuilt(image):
  image[ image > 0 ] = 255
  image[ image <= 0 ] = 0

  thinned = cv2.ximgproc.thinning(np.array(image, dtype = np.uint8), thinningType=cv2.ximgproc.THINNING_GUOHALL)

  return np.expand_dims(thinned, axis=-1)



# ---- IMAGE ANALYSIS ---- #



# --------- IMAGE ANALYSIS ENTRY POINT ------


def are_images_closed(images, silent_mode = True):

    run_results_path_following = []
    run_results_floodFILL = []
    proper_images_arr = []

    for image_arr in images[:]:

        original_image = image_arr
        #flood_fill_image = image_arr
        # Collapse values to 1 or -1
        #flood_fill_image[ flood_fill_image > 0 ] = 1
        #flood_fill_image[ flood_fill_image <= 0 ] = -1

        # Skeleton images
        if True:
          #image_arr = extract_skeleton(image_arr)
          image_arr = np.array(extract_skeleton_inbuilt(image_arr), dtype = np.float32)
          

        
        # Collapse values to 1 or -1
        image_arr[ image_arr > 0 ] = 1
        image_arr[ image_arr <= 0 ] = -1

        # Prepare the new mask
        mask = np.zeros_like(image_arr)
        mask[:,:] = -1

        # Get a starting point to follow a path
        starting_point, mask = follow_a_path_starting_point(image_arr, mask)
        
        # Reset locations visited
        location_visited = {}

        new_starting_point, mod_image, mask, location_visited, is_track_closed = follow_a_path( starting_point, image_arr, mask, location_visited, starting_point, standard_mode = True, silent_mode = True )
        
        # check wheter there is another starting point (images with artifacts)
        starting_point, _ = follow_a_path_starting_point(mod_image, mask)

        is_the_image_valid = is_track_closed and not starting_point
        # Drive through the track in reverse mode
        #if not is_track_closed:
        #  new_starting_point, mask, location_visited, is_track_closed = follow_a_path( new_starting_point, image_arr, mask, location_visited, starting_point, standard_mode = True, silent_mode = True )

        run_results_path_following.append(is_the_image_valid)

        if is_the_image_valid:
            proper_images_arr.append(original_image)


        if not silent_mode:
            print(is_the_image_valid)
            display_multiple_image([denormalize_image(image_arr), denormalize_image(mask)])
        
        if is_the_image_valid and not silent_mode:
            display_multiple_image([denormalize_image(image_arr), denormalize_image(mask)])


        # Flood Fill method
        #is_track_closed = is_curve_closed_floodFILL(flood_fill_image, silent_mode = True)
        #run_results_floodFILL.append(is_track_closed)
    if not silent_mode:
      print(f"Ending Results \n\nTracks completed by path following {np.sum(run_results_path_following)} of {len(run_results_path_following)}")
      #print(f"Tracks closed by floodFILL {np.sum(run_results_floodFILL)} of {len(run_results_floodFILL)}")
    
    return proper_images_arr





#-------





# Get a starting point to follow a path
def follow_a_path_starting_point(image, mask):
  
  for i in range(image.shape[0]):
      for j in range(image.shape[1]):

          if image[i,j] == 1:
              mask[i,j] = 1

              starting_point = structures.Coordinate(i, j)
              return starting_point, mask
        
  return False, mask


# Check wheter a closed path exists
def follow_a_path( starting_point, image_arr, mask, location_visited ,goal_point , standard_mode = True, silent_mode = False):
  
  looping = 1
  #location_visited[f"{starting_point.x}{starting_point.y}"] = 1
  new_starting_point = starting_point
  #new_starting_point = Coordinate(90,60)
  local_search_range = 1
  #new_starting_point.x
  cycles = 1
  # Wheter the track is closed
  is_track_closed = False

  while(looping):

    new_point_found = False
    x, y = new_starting_point.x, new_starting_point.y

    #steps = [ [-1,0], [-1, 1], [0,1], [1,1], [0,1], [1,-1],[0,-1], [-1,-1] ]
    #step = [i for i in range(-4, 5)]
    
    #steps = [ [i, j] for i in step for j in step ]
    #random.shuffle(steps)
    

    #steps = [ [-local_search_range,0], [-local_search_range, local_search_range], [0,local_search_range], [local_search_range,local_search_range], [0,local_search_range], [local_search_range,-local_search_range],[0,-local_search_range], [-local_search_range,-local_search_range] ]

    #print(local_search_range)
    step = [i for i in range(-local_search_range, local_search_range+1)]
    
    
    
    #random.shuffle(steps)

    #Right to Left
    if standard_mode:
      steps = [ [i, j] for i in step for j in step ]
    else:
      steps = [ [i, j] for i in reversed(step) for j in reversed(step) ]
    
    if False:
      steps_a = [ [i, j] for i in step for j in step ]
      steps_b = [ [i, j] for i in reversed(step) for j in reversed(step) ]
      steps = steps_a + steps_b


    for x_step, y_step in steps:

    #step = [0, 1, -1, 2, -2, 3, -3]

    #for x_step in step:

        if new_point_found:
          # Shrink local search
          local_search_range = 1

          #break
        
    #  for y_step in step:

        if (x_step == 0) and (y_step== 0):
          continue

        try:
          a = image_arr[x+x_step, y + y_step] == 0
        except IndexError as e:
          # Out of bound
          continue
        
        

        
        if (  image_arr[x+x_step, y + y_step] == 1  ):
            new_x = x + x_step
            new_y =  y + y_step

            #print(f"Checking {new_x, new_y}")
            if f"{new_x}{new_y}" not in location_visited:
              
              location_visited[f"{new_x}{new_y}"] = 1

              new_starting_point = structures.Coordinate(new_x, new_y)
              
              #print(f"New Starting Point {new_x, new_y}")
              
              mask[new_x, new_y] = min(0.01 * cycles, 1)

              # Remove the visited locations from the original image
              image_arr[new_x, new_y] = -1

              new_point_found = True

              #location_visited
              #break
          
          
            # Check win condition
            if cycles >30: #In order to avoid checking at the start
              if ( abs(goal_point.x - new_x) <=2 ) and ( abs(goal_point.y - new_y) <=2 ):

                if not silent_mode:
                  print("SUCCESS")
                  print(f"Starting point was : {goal_point.x, goal_point.y}")
                  print(f"Ending pointo x:{new_x} y:{new_y}")
                
                is_track_closed = True

                looping = False
                break


        '''
        if (starting_point.x == new_x) and (starting_point.y == new_y) and (looping > 5):
          print("SUCCESS")
          print(f"Starting point was : {new_x, new_y}")
          looping = False
          break
        '''
            
            
            
      
      
          #print("already visited")

    if not new_point_found:
      
      # No new point is reachable
      if local_search_range == 10:
        #print("Looping vanelessly! Exiting...")
        break 

      local_search_range = min(local_search_range+1, 10)


    cycles+=1
    if cycles == 400:
      if not silent_mode:
        print("Counter exceeded")
        print(f"Lasto pointo x:{new_x} y:{new_y}")
      break

    if (cycles % 75 == 0) and (not silent_mode):
      display_multiple_image([denormalize_image(image_arr), denormalize_image(mask)])
    
  return new_starting_point, image_arr , mask, location_visited, is_track_closed

# Flood FIll method

def is_curve_closed_floodFILL(image , silent_mode = False):
  # load and convert to grayscale
  #img = cv2.imread('unclosed.png')
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #img = images[0]
  img = np.array(image, dtype = np.uint8)
  # floodFill
  img = cv2.bitwise_not(img)

  if not silent_mode :
      print("BITWISE NOT")
      display_multiple_image( [np.reshape(img, [128, 128 , 1] )])

  mask = np.zeros((np.array(img.shape)+2), np.uint8)
  _ = cv2.floodFill(img, mask, (0,0), (255))

  if not silent_mode :
      print(f"\n MASK SHAPE : {mask.shape}")
      display_multiple_image( [np.reshape(mask, [130, 130 , 1] )])
  # erode
  img = cv2.bitwise_not(img)

  if not silent_mode :
      print("BITWISE NOT")
      display_multiple_image( [np.reshape(img, [128, 128 , 1] )])

  #print("\nasdasdad\n", img.shape, "\nasdads")

  img = cv2.erode(img, np.ones((3,3)))

  img.shape
  img = np.reshape(img, [128, 128 , 1] )
  
  if not silent_mode :
      print("EROSION")
      display_multiple_image( [img])

  if img.sum() > 0:
      if not silent_mode :
          print('Closed curve detected')
      return True
  else:
      if not silent_mode :
          print('Closed curve was not detected')
      return False


''' CONTORUING METHOD NOT TESTED

src = cv2.imread('test.png', cv2.IMREAD_COLOR)

#Transform source image to gray if it is not already
if len(src.shape) != 2:
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
else:
    gray = src

ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]

for i, c in enumerate(contours):
    if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
        cv2.drawContours(src, contours, i, (0, 0, 255), 2)
    else:
        cv2.drawContours(src, contours, i, (0, 255, 0), 2)
#write to the same directory
cv2.imwrite("result.png", src)

'''