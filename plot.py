import pandas as pd 
import matplotlib.pyplot as plt 
#from scipy.misc import imread
#import matplotlib
from tkinter import *
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox, AnchoredOffsetbox
import matplotlib.image as mpimg
import os 
import numpy as np 
#from scipy.misc import imread
#matplotlib.use('TkAgg')
csv_pure = pd.read_csv("/home/bwtseng/Downloads/model_compression/model_save/Pre_test_robustness_mobilenet_v1_imagenet_log_2020.06.23-065106/Pre_test_robustness.csv")
csv_noise = pd.read_csv("/home/bwtseng/Downloads/model_compression/model_save/Test_Robustness_mobilenet_v1_imagenet_log_2020.06.23-065103/Test_Robustness.csv")
csv_channel = pd.read_csv('/home/bwtseng/Downloads/model_compression/model_save/Channel_robustness_mobilenet_v1_imagenet_log_2020.06.24-181239/Channel_robustness.csv')
image_path = '/home/bwtseng/Downloads/model_compression/model_save/Plot_mobilenet_v1_imagenet_log_2020.06.23-163928/'
image_noise_factor = [0, 0.5, 1, 1.5, 2, 2.5, 3]
acc_pure = csv_pure[["Accuracy"]]
acc_noise = csv_noise[["Accuracy"]]
acc_channel = csv_channel[['Accuracy']]
noise_factor = csv_pure[["Noise_factor"]]
axis_list = [(0.120464, 68.564), (0.6288, 64.2141), (1.08246, 56.9124), (1.61425, 55.2133), 
            (2.10493, 55.2133), (2.60869, 55.2133), (3.035, 55.2133)]


for i in range(len(axis_list)):
    temp = axis_list[i][0]- 0.1
    temp_1 = axis_list[i][1] - 9
    axis_list[i] = (temp, temp_1)
fig, ax = plt.subplots()
ax.plot(noise_factor, acc_pure, label="Pretrained")
ax.plot(noise_factor, acc_noise, label="Fine grained sparsity")
ax.plot(noise_factor, acc_channel, label="Channel sparsity")

for i in range(len(image_noise_factor)):
    temp = mpimg.imread(os.path.join(image_path, str(i)+'.png'))
    #print(temp)
    #temp = plt.imread(os.path.join(image_path, str(0)+'.png'))
    #temp = np.arange(100).reshape((10, 10))
    imagebox = OffsetImage(temp, zoom=0.15)
    #imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, (axis_list[i][0], axis_list[i][1]), 
                        xycoords='data', frameon=False, pad=0)
    #ab = AnchoredOffsetbox(loc=3, child=imagebox, frameon=False, pad=0)#(0.5, 0.5))#,  xybox=(0., -16.), frameon=False,
                            #xycoords='data',  boxcoords="offset points", pad=0)
    ax.add_artist(ab)
plt.xlabel("Multiplication Factor")
plt.ylabel("Accuracy (%)")
#newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE', zorder=-1)
plt.legend()
plt.grid()
#plt.draw()
plt.savefig('results.png', bbox_inches='tight')
plt.show()