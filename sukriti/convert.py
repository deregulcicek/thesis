from PIL import Image
from skimage import data, io, filters
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets


upath = "/Users/kunthar/development/gulcicek/unet/data/train/label/"
n=10  #number of images

for i in range (1,n+1):
	path=str(i)+'.jpg'
	img= io.imread(upath +path)
	#Subsection of the image
	img=img[52:308,52:308]
	print(img.shape)
	path=str(i)+'.tif'
	io.imsave(upath +path,img)
