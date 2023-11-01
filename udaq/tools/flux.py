import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
#hep.style.use('LHCb2')

def flux(number_of_muons,area,top_solid_angle,alpha,epsilon,time):
    muon_flux = number_of_muons/(top_solid_angle*area*alpha*epsilon*time)
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

efficiencies = {'A':54.59/100,'B':76.36/100,'C':90.27/100,'D':20.02/100,'E':50.17/100,'F':61.28/100,'G':87.16/100,'H':2.24/100}     #decimal
efficiencies_uncertainty = {'A':0.71,'B':0.72,'C':0.90,'D':0.58,'E':1.03,'F':1.13,'G':3.20,'H':0.30} #in percentages

epsilon = 21.5509/100 #efficiencies['A']*efficiencies['C']*efficiencies['E']*efficiencies['G'] #efficiency correction
number_of_muons = df.index.size     #number of triggers
run_time = 19000 #seconds
scintillator_area = 0.6**2 #m^2
top_solid_angle = (np.pi**2)/2 #solid angle top scintillator
alpha = 0.5 # rough proportion of top scintilaltor solid angle as seen from bottom scintillator, provided by supervisor

#error
N_error = 0 #integer (if any?)
A_error = 0  #in m^-2
omega_error = 0  #in str (if any?)
alpha_error = 0  #error is derived from measurements of apparatus? 
epsilon_error = 0.6314/100    #propogation of uncertainties on the efficiencies (not in percentage error of a percentage!)
t_error = 0  #time error (if any?)

muonflux_error = error(number_of_muons, scintillator_area, top_solid_angle, alpha, epsilon, run_time, N_error, A_error, omega_error, alpha_error, epsilon_error, t_error)

print("Muon flux = (",flux(number_of_muons,scintillator_area,top_solid_angle,alpha,epsilon,run_time),"pm",muonflux_error,") s^-1 str^-1 m^-2")

#need to find uncertainties on N, A, omega, alpha, epsilon and time t.