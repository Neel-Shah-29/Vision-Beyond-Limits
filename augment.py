# # /home/ppspr/Downloads/newImages/8-o-clock-1453175.jpg
import numpy as np
import os
from PIL import Image

def augment(input_folder):
	for image_name in os.listdir(input_folder):
		image_path = os.path.join(input_folder, image_name)
		input_image = np.array(Image.open(image_path))
		image_path = image_path.replace('.png','')
		
		output_path = os.path.join(output_folder_90,image_name+'_90.png')
		Image.fromarray(np.rot90(input_image)).save(output_path)

		output_path = os.path.join(output_folder_180,image_name+'_180.png')
		Image.fromarray(np.rot90(input_image, 2)).save(output_path)
		
		output_path = os.path.join(output_folder_270,image_name+'_270.png')
		Image.fromarray(np.rot90(input_image, 3)).save(output_path)

input_folder = '/home/ppspr/code/python/vbl_data/orginal_images'
output_folder_90 = '/home/ppspr/code/python/vbl_data/augmented_data/data_90/images_90'
output_folder_180 = '/home/ppspr/code/python/vbl_data/augmented_data/data_180/images_180'
output_folder_270 = '/home/ppspr/code/python/vbl_data/augmented_data/data_270/images_270'

augment(input_folder)

input_folder = '/home/ppspr/code/python/vbl_data/original_mask'
output_folder_90 = '/home/ppspr/code/python/vbl_data/augmented_images/data_90/masks_90'
output_folder_180 = '/home/ppspr/code/python/vbl_data/augmented_images/data_180/masks_180'
output_folder_270 = '/home/ppspr/code/python/vbl_data/augmented_images/data_270/masks_270'

augment(input_folder)
