import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import random
import mplhep as hep
import os
#hep.style.use('LHCb2')

def flux(number_of_muons,area,solid_angle,alpha,epsilon,time):
    muon_flux = number_of_muons/(solid_angle*area*alpha*epsilon*time)
    return(muon_flux)

def error(N,A,omega,alpha,epsilon,t,N_error,A_error,omega_error,alpha_error,epsilon_error,t_error):
    par_N = 1/(omega*A*t*alpha*epsilon)
    par_omega = -N/((omega**2)*A*t*alpha*epsilon)
    par_A = -N/((A**2)*omega*t*alpha*epsilon)
    par_t = -N/((t**2)*A*omega*alpha*epsilon)
    par_alpha = -N/((alpha**2)*A*t*omega*epsilon)
    par_epsilon = -N/((epsilon**2)*A*t*alpha*omega)

    fluxerror = np.sqrt( ((par_N**2)*(N_error**2)) + ((par_omega**2)*(omega_error**2)) + ((par_A**2)*(A_error**2)) + ((par_t**2)*(t_error**2)) + ((par_alpha**2)*(alpha_error**2)) + ((par_epsilon**2)*(epsilon_error**2)) )
    return(fluxerror)

channels = ['A','B','C','D','E','F','G','H']
df = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Flux_CSV/Run0000.csv' )   #will need changing if not running on laptop

# Initialising variables
efficiencies = {'A':54.59/100,'B':76.36/100,'C':90.27/100,'D':20.02/100,'E':50.17/100,'F':61.28/100,'G':87.16/100,'H':2.24/100}     #decimal
efficiencies_uncertainty = {'A':0.71,'B':0.72,'C':0.90,'D':0.58,'E':1.03,'F':1.13,'G':3.20,'H':0.30} #in percentages
a=40.000e-2    # Scintillator length in m
b=42.000e-2    # Scintillator width in m
diag = np.sqrt(a**2 + b**2)
scintillator_area = a*b
d = 0.22 # In m, pm 0.5mm - height from top of C scintillator to top of A scintillator
k= d/diag # Used in error prop later
theta_max = np.pi/2 - np.arctan(k) # theta_max for solid angle
epsilon = 0.215509 #efficiencies['A']*efficiencies['C']*efficiencies['E']*efficiencies['G'] #efficiency correction
number_of_muons = df.index.size # Number of triggers 
#print("\nN=",number_of_muons,"\n")  
run_time = 19000 # Seconds
deadtime = number_of_muons*(1*10**(-3))    # Dead time of 10ms per trigger event (captures per block = 1)
solid_angle = - 2*np.pi*(((np.cos(theta_max))**3)/3 - ((np.cos(0))**3)/3) # Solid angle top scintillator, integral of (cos^2(theta)*sin(theta)), limits 0 - theta_max
alpha = 0.661721657 # Rough proportion of top scintilaltor solid angle as seen from bottom scintillator (averaged over 10 monte carlo runs)

# Errors
d_err=0.5e-3    # In metres. See section 5.2 in onenote 
a_err = b_err = 0.0005e-2   # Manufacturer uncertainty on a and b in m
theta_error = 100*(( np.sqrt( ((d_err**2)*(1/(np.sqrt(a**2 + b**2)))**2) + ((a_err**2)*(-(d*a)/((a**2 + b**2)**(3/2)))**2) + ((b_err**2)*(-(d*b)/((a**2 + b**2)**(3/2)))**2)  ) )/k)  # Spits out percentage error
print("\ntheta_max = (", theta_max, "pm", theta_max*theta_error/100, ") rad")
N_error = 0 # Error on N is adopted into the epsilon efficiency correction 
A_error = np.sqrt(((b**2)*(a_err**2)) + ((a**2)*(b_err**2)))  #in m^-2 
omega_error = (theta_error/100)*solid_angle  # In str (derived from a,b and d)
print("omega = (",solid_angle,"pm", omega_error,") str")
alpha_error = 0.007132347409294789  # Error is derived from std error on 10 monte carlo simulation runs
epsilon_error = 0.006314    # Propogation of uncertainties on the efficiencies (not in percentage error of a percentage!)
t_error = deadtime  # Derived from dead time

print("N=",number_of_muons)
print("area = (",scintillator_area, "pm",A_error,") m^-2")
print("alpha =",alpha,"pm",alpha_error)
print("epsilon =",epsilon,"pm",epsilon_error)
print("runtime = (",run_time,"pm",t_error, ") s")

muonflux_error = error(number_of_muons, scintillator_area, solid_angle, alpha, epsilon, run_time, N_error, A_error, omega_error, alpha_error, epsilon_error, t_error)
print("Muon flux = (",flux(number_of_muons,scintillator_area,solid_angle,alpha,epsilon,run_time),"pm",muonflux_error,") s^-1 str^-1 m^-2")
print("Muon flux = (",flux(number_of_muons,scintillator_area,solid_angle,alpha,epsilon,run_time)*60,"pm",muonflux_error*60,") minuites^-1 str^-1 m^-2")