import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
hep.style.use('LHCb2')

#os.system("python ./udaq/src/udaq.py ./udaq/tools/udaq per channel.config")
#doesn't work - need to use anaconda powershell to run experiment and use this file to analyse it

channels = ["A","B","C","E","F","G"]    #missing out D and H as they have range 0.1 and range 0.5 datasets 

for channel in channels:

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

    plt.errorbar(thresholds,lengths/time,yerr=np.sqrt(lengths)/time)
    plt.xlabel('Threshold (V)',fontsize=32)
    #plt.yscale("log")  
    plt.ylabel('Trigger Rate (Hz)',fontsize=32)
    plt.xlim(0, 0.3)
    plt.savefig('rate_thresh_%s_0000.pdf' % channel)                            
    plt.clf()
    plt.cla()
    # plt.errorbar(thresholds,lengths/time,yerr=np.sqrt(lengths)/time)
    # plt.xlabel('Threshold (V)',fontsize=32)
    # plt.yscale("log")  
    # plt.ylabel('Trigger Rate (Hz)',fontsize=32)
    # plt.savefig('rate_thresh_log_%s_0000.pdf' % channel)                       