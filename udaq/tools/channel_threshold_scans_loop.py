import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
hep.style.use('LHCb2')

#os.system("python ./udaq/src/udaq.py ./udaq/tools/udaq per channel.config")
#doesn't work - need to use anaconda powershell to run experiment and use this file to analyse it

channels = ["A","B","C","D","E","F","G","H"]    #missing out D and H as they have range 0.1 and range 0.5 datasets 

threshold_y = [0,0,0,0,0,0,0,0]
threshold_x = [0.1200, 0.0900, 0.1400, 0.0500, 0.0800, 0.0400, 0.0700, 0.0300]
colour_array = [0,0,0,0,0,0,0,0]
fs = "35" # Font size


th_i = -1
for channel in channels:
    th_i += 1
    if (channel == "D" or channel == "H"):
        df = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/threshold_analysis/Channel_%s/Run0000_range0.5.csv' % channel)    #change dir
    else:
        df = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/threshold_analysis/Channel_%s/Run0000.csv' % channel)    #change dir

    time = 600 #in seconds
    N=250
    #array of thresholds to compare to
    thresholds = np.linspace(0,1.5,N)

    #filter thresholds from thresholds array
    lengths  = np.zeros(N)
    for i in range(0,N):    
        df_filter = df.query(f"pulse_height_%s>{thresholds[i]}" % channel)                  
        lengths[i] = len(df_filter)
    
    yvals = lengths/time
    yerrs = 1*np.sqrt(lengths/time)     #95% confidence - poisson estimate of error sqrt(lambda/time) (lambda is the mean number of events within a given interval of time or space)
    

    plot = plt.errorbar(thresholds,yvals, yerr = yerrs, linewidth = 2, label= ("Channel %s" % channel),elinewidth=1,capthick=1) #add:   ,marker='.',markersize=5)    for markers
    plt.xlabel('Threshold (V)',fontsize=fs)
    #plt.yscale("log")  
    plt.ylabel('Trigger Rate (Hz)',fontsize=fs)
    plt.xlim(0, 0.2)
    plt.ylim(-15,300)

    xthresh = threshold_x[th_i]
    difference_array = np.absolute((thresholds)-xthresh)
    # find the index of minimum element from the array
    index = difference_array.argmin()
    yval = yvals[index]
    threshold_y[th_i] = yval
    #print(threshold_y)

    colour_array[th_i] = plot[0].get_color()
    print(colour_array)

#correcting positions of threshold markers (some are too low/high)
base_C = threshold_y[2]
threshold_y[2] = base_C -0.2

base_D = threshold_y[3]
threshold_y[3] = base_D -0.3

base_E = threshold_y[4]
threshold_y[4] = base_E -0.5

base_F = threshold_y[5]
threshold_y[5] = base_F +0.7

base_G = threshold_y[6]
threshold_y[6] = base_G +0.7

# #x corrections

# xbase_A = threshold_x[0]
# threshold_x[0] = xbase_A + 0.0008

# xbase_B = threshold_x[1]
# threshold_x[1] = xbase_B + 0.00075

# xbase_C = threshold_x[2]
# threshold_x[2] = xbase_C -0.0009

# xbase_D = threshold_x[3]
# threshold_x[3] = xbase_D - 0.0015

# xbase_E = threshold_x[4]
# threshold_x[4] = xbase_E - 0.0015

# xbase_F = threshold_x[5]
# threshold_x[5] = xbase_F + 0.0025

# xbase_G = threshold_x[6]
# threshold_x[6] = xbase_G + 0.0025


plt.scatter(threshold_x,threshold_y,s=250,marker="x",zorder=10,c=colour_array)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
# plt.grid(linestyle='--',which='both')
# ax = plt.gca()
# ax.minorticks_on()
plt.legend(fontsize=fs)
plt.show()
#plt.savefig('thresholds.pdf')                            
    # plt.clf()
    # plt.cla()
    # plt.errorbar(thresholds,lengths/time,yerr=np.sqrt(lengths)/time)
    # plt.xlabel('Threshold (V)',fontsize=32)
    # plt.yscale("log")  
    # plt.ylabel('Trigger Rate (Hz)',fontsize=32)
    # plt.savefig('rate_thresh_log_%s_0000.pdf' % channel)                       