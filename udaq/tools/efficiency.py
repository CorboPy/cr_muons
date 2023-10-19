import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
#hep.style.use('LHCb2')

threshs_dict = {'A':0.12,'B':0.09,'C':0.14,'D':0.06,'E':0.08,'F':0.04,'G':0.07,'H':0.06}    #dict of thresholds set for each channel 

dfA_2 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_A/Run0000.csv' )
dfA_3 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_A/Run0000.csv' )

dfB_2 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_B/Run0000.csv' )
dfB_3 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_B/Run0000.csv' )

dfC_2 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_C/Run0000.csv' )
dfC_3 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_C/Run0000.csv' )

dfD_2 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_D/Run0000.csv' )
dfD_3 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_D/Run0000.csv' )

dfE_2 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_E/Run0000.csv' )
dfE_3 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_E/Run0000.csv' )

dfF_2 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_F/Run0000.csv' )
dfF_3 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_F/Run0000.csv' )

dfG_2 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_G/Run0000.csv' )
dfG_3 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_G/Run0000.csv' )

dfH_2 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_H/Run0000.csv' )
dfH_3 = pd.read_csv('C:/Users/alexc/Documents/#Uni/Physics with Astrophysics/YR 3/YR3 Labs/CR Experiment/Efficiency/Channel_H/Run0000.csv' )

dfdict={'A2':dfA_2,'A3':dfA_3,'B2':dfB_2,'B3':dfB_3,'C2':dfC_2,'C3':dfC_3,'D2':dfD_2,'D3':dfD_3,'E2':dfE_2,'E3':dfE_3,'F2':dfF_2,'F3':dfF_3,'G2':dfG_2,'G3':dfG_3,'H2':dfH_2,'H3':dfH_3}

channels = ['A','B','C','D','E','F','G','H']

for i in range(len(channels)):
    channel = channels[i]   #specific channel
    df2 = dfdict[str(channel+'2')]  #twofold coincidence df
    df3 = dfdict[str(channel+'3')]  #threefold coincidence df

    num_triggers_2 = df2.index.size     #number of triggers on ref channels (twofold)
    num_triggers_3 = df3.index.size     #number of triggers on ref channels (threefold)

    df_filter_2 = df2.query(f"pulse_height_%s>{threshs_dict[channel]}" % channel)   #filtering twofold df for sucessful counts on test channel
    success_num_2 = df_filter_2.index.size      #counting num of sucessful counts

    df_filter_3 = df3.query(f"pulse_height_%s>{threshs_dict[channel]}" % channel)  #filtering threefold df for sucessful counts on test channel
    success_num_3 = df_filter_3.index.size      #counting num of sucessful counts

    #for twofold
    twofold_efficiency=(success_num_2/num_triggers_2)*100
    threefold_efficiency = (success_num_3/num_triggers_3)*100