import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
hep.style.use('LHCb2')

#os.system("python ./udaq/src/udaq.py ./udaq/tools/udaq per channel.config")
#doesn't work - need to use anaconda powershell and use this file to analyse it

df = pd.read_csv('C:/Users/xz21360/CR code/CR_MSci_Lab/Run0000.csv')


time = 600 #in seconds
N=250
#array of thresholds to compare to
thresholds = np.linspace(0,1.5,N)


#filter thresholds from thresholds array
lengths  = np.zeros(N)
for i in range(0,N):    
    df_filter = df.query(f"pulse_height_A>{thresholds[i]}")
    lengths[i] = len(df_filter)

plt.errorbar(thresholds,lengths/time,yerr=np.sqrt(lengths)/time)
plt.xlabel('Threshold (V)',fontsize=32)
#plt.yscale("log")  
plt.ylabel('Trigger Rate',fontsize=32)
plt.savefig('rate_thresh.pdf')
plt.clf()
plt.cla()
plt.errorbar(thresholds,lengths/time,yerr=np.sqrt(lengths)/time)
plt.xlabel('Threshold (V)',fontsize=32)
plt.yscale("log")  
plt.ylabel('Trigger Rate',fontsize=32)
plt.savefig('rate_thresh_log.pdf')