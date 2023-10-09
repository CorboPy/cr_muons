import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
hep.style.use('LHCb2')

os.system("python ./udaq/src/udaq.py ./udaq/tools/udaq_single.config")

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




""" for ch in ['A','B','C','D','E','F','G','H']:
    h, bins = np.histogram(df[f'pulse_height_{ch}'],range=[0,1],bins=100)
    hep.histplot(h,bins,yerr=np.sqrt(h),histtype='step',label=f'Channel {ch}')

plt.legend()
plt.xlabel('Peak of channel (V))',fontsize=32)
plt.ylabel('Counts',fontsize=32)
plt.savefig('peak_distbn.pdf')

thA = 0.1
thB = 0.1
thC = 0.1
thD = 0.01
thE = 0.1
thF = 0.1
thG = 0.1
thH = 0.01

# run for trigger 1 save events/time , run for trigger 2 .......
# run for long then , plot for pulse hight over x 
df_BDF = df.query(f"pulse_height_B>{thB} & pulse_height_D>{thD} & pulse_height_F>{thF} & pulse_height_H<{thH}")
df_BDH = df.query(f"pulse_height_B>{thB} & pulse_height_D>{thD} & pulse_height_H>{thH} & pulse_height_F<{thF}")
df_BFH = df.query(f"pulse_height_B>{thB} & pulse_height_F>{thF} & pulse_height_H>{thH} & pulse_height_D<{thD}")
df_DFH = df.query(f"pulse_height_D>{thD} & pulse_height_F>{thF} & pulse_height_H>{thH} & pulse_height_B<{thB}")
df_BDFH = df.query(f"pulse_height_D>{thD} & pulse_height_F>{thF} & pulse_height_H>{thH} & pulse_height_B>{thB}")

print(f"Number of BDF triggers and !H: {df_BDF.shape[0]}")
print(f"Number of BDH triggers and !F: {df_BDH.shape[0]}")
print(f"Number of BFH triggers and !D: {df_BFH.shape[0]}")
print(f"Number of DFH triggers and !B: {df_DFH.shape[0]}")
print(f"Number of BDFH triggers: {df_BDFH.shape[0]}")
print(f"Total number of triggers: {df.shape[0]}") """