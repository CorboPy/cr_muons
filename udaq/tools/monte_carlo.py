import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import mplhep as hep
hep.style.use('LHCb2')

# Number of random points to sample
N = 25
N_2D = N**4

# Initialize variables
theta_max = np.pi/2 - np.arctan(0.22/(np.sqrt(0.409878**2 + 0.409878**2)))      # Trig calc for theta max assuming square scintillators of same area
h=0.22 #m

xvals = np.linspace(-0.409878/2,0.409878/2,N)     #0.409878 metres approximating a square with same area
yvals = np.linspace(-0.409878/2,0.409878/2,N) 
theta_vals = np.zeros(N)
phi_vals = np.zeros(N)

# Setting up arrays of theta and phi values
for j in range(len(phi_vals)):
     phi_vals[j] = np.random.uniform(0,2*np.pi)  # Azimuthal angle (φ)
for i in range(len(theta_vals)):
     theta_vals[i] = np.random.uniform(0,theta_max)  # Polar angle (θ)

count = 0
index = 0
# Perform Monte Carlo simulation
for x in xvals:
    index=index+1
    print(round(index/N * 100,1), f"% complete")
    for y in yvals:

    # Check x and y projections onto scintillator plate at distance h both lie within the bounds of the square
        
        for theta in theta_vals:
            for phi in phi_vals:
                 
                if (((h*np.tan(theta)*np.sin(phi))+x) <= abs(max(xvals))) and ((((h*np.tan(theta)*np.cos(phi)))+y) <= abs(max(yvals))):
                    count=count+1

print("Successes:", count,". Failuires:",N_2D-count)    # N_2D instead of N because count goes like N**4
print("alpha=", count/N_2D, "pm", np.sqrt(((count/N_2D) * (1 - (count/N_2D)) )/ N_2D))

# Plotting example setting vector origin to (0,0) on lower scintillator plate
x_arr = xvals + (h*np.tan(theta_vals)*np.sin(phi_vals))
y_arr = yvals + (h*np.tan(theta_vals)*np.cos(phi_vals))

font = {'fontname':'Calibri'}
aperture = patch.Rectangle((-0.409878,-0.409878),2*0.409878,2*0.409878,fill=False,color='red')

plt.scatter(x_arr,y_arr,marker="x",color='black',label="vector trajectory")
plt.xlabel("$x (m)$")
plt.ylabel("$y (m)$")
plt.legend(loc="upper right",frameon=True)
plt.text(-0.4,0.36,"upper scintillator target",fontsize="xx-large",color='red',weight='bold',**font)
plt.title("Example Monte Carlo method for N=%s vectors with origin (0,0) on lower scintillator" % N,fontsize="x-large",weight='bold')
plt.locator_params(axis='both', nbins=5) 

ax = plt.gca()
ax.add_patch(aperture)
ax.set_xlim(-0.7,0.7)
ax.set_ylim(-0.7,0.7)

plt.show()
