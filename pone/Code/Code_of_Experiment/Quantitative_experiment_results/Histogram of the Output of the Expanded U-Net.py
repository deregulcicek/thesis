from PIL import Image
import numpy as np
import cv2 as cv2
from skimage import data
import matplotlib.pyplot as plt

img = cv2.imread('H:/Data/Code/Code_of_Experiment/Quantitative_experiment_results/Experimental_Data_and_Results/3channel_3class_w_and_b_benign/predict/0.jpg')
plt.figure("breast")
ar = img[:,:,0].flatten()
plt.hist(ar, bins = 256, normed= 1, facecolor = 'r', edgecolor = 'r')
ag = img[:,:,1].flatten()
plt.hist(ag, bins = 256, normed = 1, facecolor = 'g', edgecolor = 'g')
ab = img[:,:,2].flatten()
plt.hist(ab, bins = 256, normed = 1, facecolor = 'b', edgecolor = 'b')
plt.show()