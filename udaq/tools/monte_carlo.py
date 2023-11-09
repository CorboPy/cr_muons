import random
import numpy as np

# Define the function cos^2(θ)
# def cos_squared(theta):
#     return math.cos(theta) ** 2

# Number of random points to sample
N = 50
N_2D = N**4

# Initialize variables to keep track of points inside the spherical region and the sum of function values
theta_max = np.pi/2 - np.arctan(0.22/(np.sqrt(0.409878**2 + 0.409878**2)))      #theta max assuming square scintillators of same area
print("theta max=", theta_max)
h=0.22 #m

xvals = np.linspace(-0.409878/2,0.409878/2,N)     #0.409878 metres approximating a square with same area
yvals = np.linspace(-0.409878/2,0.409878/2,N) 
theta_vals = np.zeros(N)
phi_vals = np.zeros(N)

for j in range(len(phi_vals)):
     phi_vals[j] = np.random.uniform(0,2*np.pi)  # Azimuthal angle (φ)
for i in range(len(theta_vals)):
     theta_vals[i] = np.random.uniform(0,theta_max)  # Polar angle (θ)
print(theta_vals)
print("\n",phi_vals)

#print((h*np.tan(theta)*np.sin(phi)))

count = 0
index = 0
# Perform Monte Carlo simulation
for x in xvals:
    index=index+1
    print(index/N * 100, f"% complete")
    for y in yvals:

    # Generate random spherical coordinates
        
        for theta in theta_vals:
            for phi in phi_vals:
                 
                if ((abs((h*np.tan(theta)*np.sin(phi))))) <= abs(max(xvals)) and ((abs((h*np.tan(theta)*np.cos(phi)))) <= abs(max(yvals))):
                        count=count+1


print("Successes:", count,". Failuires:",N_2D-count)
print("alpha=", count/N_2D)
