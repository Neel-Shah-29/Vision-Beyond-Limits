%reload_ext autoreload
%autoreload 2
%matplotlib inline

from skimage.draw import line, polygon, circle, ellipse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import skimage.io
import json
# To read files in a directory
from os import listdir
from os.path import isfile, join

# To load wkt; this is a specific method in shapely library 
from shapely.wkt import loads
images_path = Path('/content/drive/MyDrive/VisionBeyondLimits/Images')

# We construct the path to label path where we want to put the mask image 
label_path = Path('/content/drive/MyDrive/tp')

# We construct the path to the json file; the json file contains coordinates of polygons
json_path  = Path('/content/drive/MyDrive/VisionBeyondLimits/Labels')

list_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

counter = 0

for img_name in list_files:
    # split the file name
    prefix_file_name = img_name.split(".")
    
    # construct the path to the image
    temp_image_path = images_path / img_name    
    
    # construct the path to the json    
    temp_json_path = json_path / (prefix_file_name[0]+".json")
    
    # read the json
    json_dict = None 
    with open(temp_json_path, 'r') as read_file:
        json_dict = json.load(read_file)  
    
    # construct the list of xy of buildings
    props_xy_list = json_dict['features']['xy']     
    
    # construct list of polygons 
    polygon_geom_list = []
    damage_list = []
    for prop in props_xy_list:
        polygon_temp = loads(prop['wkt'])
        polygon_geom_list.append(polygon_temp)
        damage_temp = prop['properties']['subtype']
        damage_list.append(damage_temp)
    
    # read the image which we want to draw the polygons
    the_image = skimage.io.imread( temp_image_path )    
    
    # Create the basic mask
    a_mask = np.ones(shape=the_image.shape[0:2], dtype=np.uint8) # original
    
    # For each polygon, draw the polygon inside the mask
    count =0
    for polygon_geom in polygon_geom_list:
        poly_coordinates = np.array(list(polygon_geom.exterior.coords))
        rr, cc = polygon(poly_coordinates[:,0], poly_coordinates[:,1], the_image.shape)
        if(damage_list[count] == 'no-damage'):
          a_mask[cc,rr] = 50
        elif(damage_list[count] == 'major-damage'):
          a_mask[cc,rr] = 100
        elif(damage_list[count] == 'un-classified'):
          a_mask[cc,rr] = 150
        elif(damage_list[count] == 'minor-damage'):
          a_mask[cc,rr] = 200
        elif(damage_list[count] == 'destroyed'):
          a_mask[cc,rr] = 250
        count+=1
    
    # Convert numpy array of the mask into an image with the help of PIL
    mask_image = Image.fromarray(a_mask)
    
    # Save the image of the mask into the "binaryLabels" folder 
    mask_image.save( label_path / (prefix_file_name[0]+".png"), format="PNG" )
    
    # For debugging purposes
    if counter % 1000 == 0:
        print("Number of images have been processed:", counter)
    counter += 1
