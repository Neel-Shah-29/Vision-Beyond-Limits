This is the intruction for running solution our code for Vision Beylong Limits, Techfest 2021-2022.

Directory:
	--VisionBeylondLimits
	|
	|--masking.py
	|
	|--augment.py
	|
	|--vbl.ipynb
	
Requirements:
	python==3.7.12
	keras==2.7.0
	tensorflow==2.7.0
	numpy==1.16.5
	opencv==4.1.2
	matplotlib==3.2.2
	sklearn==1.0.1
	glob
	re

Before you start you need to have Images(contains .png file) and Labels(contain .json file) directory.
To augment data you can run augment.py, it will save images rotated by 90°, 180° and 270°.

After installing all packages and libraries mentioned in requirements:
1. Run masking.py to create mask images. 
	-Enter path of Images(containg .png), Label(containg .json) and Mask where masked images wil be
	stored. 
2. Open vbl.ipynb.
	-Run all cells in sequencial order
	-Enter paths of images and masked images wherever specified.
