import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv

def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append(float(row[2]))
        x.append(float(row[1]))
    return x,y
mpl.rcParams['font.family'] = 'STSong'
mpl.rcParams['font.sans-serif'] = 'NSiSun,Time New Roman'

plt.figure()
x2,y2=readcsv("H:/Data/Code/Code_of_Experiment/Fast_Convergence_of_Expanded_U-Net_Training/Model_and_Results/logmoid2/loss2.csv")
plt.plot(x2, y2, color='red', label='Dilation BCE')

x1,y1=readcsv("H:/Data/Code/Code_of_Experiment/Fast_Convergence_of_Expanded_U-Net_Training/Model_and_Results/logmoid1/loss1.csv")
plt.plot(x1, y1, color='g', label='Classics BCE')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.ylim(0, 1.5)
plt.xlim(0, 610)
plt.xlabel('Epoch',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend(fontsize=16)
plt.show()