import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
#hep.style.use('LHCb2')

threshs = {'A':0.12,'B':0.09,'C':0.14,'D':0.06,'E':0.08,'F':0.04,'G':0.07,'H':0.06}    #dict of thresholds set for each channel 
# channels = ['A','B','C','D','E','F','G','H']

df = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Lifetime/Run0000.csv' )

df_filter1= df.query(f"pulse_height_C>{threshs['C']} & pulse_height_D<{threshs['D']}")
                    #C above threshold AND D below threshold             
                    #C because we want electrons that decay downwards. Also making sure D hasn't detected a muons passing through

#separate np arrays containing B and C times from df
times_B = df_filter1[['time_B']].to_numpy()    
times_C = df_filter1[['time_C']].to_numpy()
#taking difference between them:
diffs = times_C-times_B
#print(diffs)

#the following code is extracting muons passed straight through and electron decays from the diff array (0.5e-6 seconds is changable and best judged from log hist of diff array)
#It is a rough distinction that separates muons passing through (<0.5e-6) and electrons decaying after muon lifetime (>0.5e-6))
muons = []
electrons = []
num_decays = 0
for event in diffs:
    if event < 0.5e-6:
        muons.append(event.tolist())
    else:
        print("Muon decay detected!")
        electrons.append(event.tolist())
        num_decays=num_decays+1

#errors
muon_std = np.std(muons)    #standard deviation on passing muon times
muon_speed_error = np.sqrt( ((-0.22/((np.average(muons))**2))**2)*(muon_std)**2 + ((1/((np.average(muons))))**2)*(0.5e-3)**2  )         #partial diff method on vbar = d/tbar (d=0.22m)

#printing info
print("\nnumber of muon -> electron decays =", num_decays,". Decay times are:",electrons,"\n")
print("average time for muons passing through =",np.average(muons),"s")
#muon speeds (limited by how fast the apparatus can register data, so this is really a very rough estimate):
print("average speed of muons passing straight through (m/s)=",(0.22/np.average(muons)),"pm",muon_speed_error)
print("average speed of muons passing straight through (c)=",((0.22/(np.average(muons)))/(299792458)),"pm",(muon_speed_error/299792458))

#fig 1 - plotting all diffs for all events (helpful when figuring out the distinction time)
f1 = plt.figure(1)
plt.hist(diffs,bins=50)
plt.yscale("log")
plt.ylabel("no. of events")
plt.xlabel("time_C - time_B")
plt.title("time_c - time_b for all events")

#fig 2 - should be a nice decaying exponential (hopefully)
f2 = plt.figure(2)
plt.hist(electrons)
plt.ylabel("no. of decays")
plt.xlabel("time_C - time_B")
plt.title("muon decay time distribution")

plt.show()