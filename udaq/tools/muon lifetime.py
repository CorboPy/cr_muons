import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
from scipy.optimize import curve_fit
hep.style.use('LHCb2')

# Negative exponential function (for best fit)
def exp_func(x, a, b):
    return a * np.exp(-b * x)

threshs = {'A':0.12,'B':0.09,'C':0.14,'D':0.05,'E':0.08,'F':0.04,'G':0.07,'H':0.06}    #dict of thresholds set for each channel 
#channels = ['A','B','C','D','E','F','G','H']

df = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Lifetime/Run0002.csv')

df_filter1= df.query(f"pulse_height_C>{threshs['C']} & pulse_height_D<{threshs['D']}")
                    # C above threshold AND D below threshold             
                    # C because we want electrons that decay downwards. Also making sure D hasn't detected a muons passing through

# Separate np arrays containing B and C times from df
times_B = df_filter1[['time_B']].to_numpy()    
times_C = df_filter1[['time_C']].to_numpy()
# Taking difference between them:
diffs = times_C-times_B
#print(diffs)

# The following code is extracting muons passed straight through and electron decays from the diff array (0.5e-6 seconds is changable and best judged from log hist of diff array)
# It is a rough distinction that separates muons passing through (<0.5e-6) and electrons decaying after muon lifetime (>0.5e-6))
muons = []
electrons = []
num_decays = 0
for event in diffs:
    if event < 0.5e-6:
        muons.append(event.tolist())
    else:
        #print("Muon decay detected!")
        electrons.append(event.tolist())
        num_decays=num_decays+1
print(len(muons))
# Errors
muon_err = np.std(muons)/np.sqrt(len(muons))    # Standard deviation/root N on passing muon times
muon_speed_error = np.sqrt( ((-0.11/((np.average(muons))**2))**2)*(muon_err)**2 + ((1/((np.average(muons))))**2)*(0.5e-3)**2  )         # Partial diff method on vbar = d/tbar (d=0.22m)

# Printing info
print("\nnumber of muon -> electron decays =", num_decays,". Decay times are:",electrons[0],electrons[1],"etc\n")
print("average time for muons passing through =",np.average(muons),"pm", muon_err,"s")
#muon speeds (limited by how fast the apparatus can register data, so this is really a very rough estimate):
print("average speed of muons passing straight through (m/s)=",(0.11/np.average(muons)),"pm",muon_speed_error)
print("average speed of muons passing straight through (c)=",((0.11/(np.average(muons)))/(299792458)),"pm",(muon_speed_error/299792458))

# Fig 1 - plotting all diffs for all events (helpful when figuring out the distinction time)
f1 = plt.figure(1)
plt.hist(diffs,bins=500)
plt.yscale("log")
plt.ylabel("no. of events")
plt.xlabel("time_C - time_B",labelpad=40)
plt.title("time_c - time_b for all events")

# Fig 2 - should be a nice decaying exponential (hopefully)
electron_arr = np.array(electrons)
f2 = plt.figure(2)
hist, bins, _ = plt.hist((electron_arr),bins=35,color='gray')

# Exponential fit
coeffs, covariance = curve_fit(exp_func, bins[:-1], hist)
x_fit = np.linspace(min(bins), max(bins), 100)
y_fit = exp_func(x_fit, *coeffs)
a, b = coeffs
# Extracting the diagonal elements of the covariance matrix
errors = np.sqrt(np.diag(covariance))
#print(covariance)

# Extracting the errors for a, b, and c
a_stddev, b_stddev = errors

#2 sigma confidence intervals
a_interval = 3 * a_stddev
b_interval = 3 * b_stddev
#c_interval = 2 * c_stddev
y_upper = exp_func(x_fit, a + a_interval, b + b_interval)
y_lower = exp_func(x_fit, a - a_interval, b - b_interval)
confidence_intervals2 = 2 * np.sqrt(np.diag(covariance))
#confidence_intervals3 = 5 * np.sqrt(np.diag(covariance))

print("Coefficient a:", a)
print("Coefficient b:", b)
#print("Coefficient c:", c)
 
print("Error on coefficient a:",a_stddev/np.sqrt(len(electron_arr)))
print("Error on coefficient b:",b_stddev/np.sqrt(len(electron_arr)))
#print("Error on coefficient c:",c_stddev/np.sqrt(len(electron_arr)))

print("\nMuon lifetime =", 1/b,"pm",(b_stddev/np.sqrt(len(electron_arr)))/b * 1/b )

# plt.plot(x_fit, y_fit, 'r-', label=f'N($t_d$) fit')
# plt.fill_between(x_fit, y_fit - confidence_intervals2[0], y_fit + confidence_intervals2[0], color='red', alpha=0.2, label='2σ CI')
# #plt.fill_between(x_fit, y_fit - confidence_intervals3[0], y_fit + confidence_intervals3[0], color='purple', alpha=0.2, label='5σ CI')
# #plt.fill_between(x_fit, y_fit * np.exp(confidence_intervals[1]), y_fit * np.exp(-confidence_intervals[1]), color='orange', alpha=0.5)
# plt.legend(fontsize="40")
plt.ylabel("N",fontsize="40")
plt.xlabel("$t_d$ ($\mu$s)",labelpad=40,fontsize="40")
# plt.xticks(fontsize="40")
# plt.yticks(fontsize="40")

# #plt.title("muon decay time distribution")

plt.show()