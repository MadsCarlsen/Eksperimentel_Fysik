import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.stats import chi2
from Fresnel_functions import *

#plt.rc('text', usetex=True)
plt.rc("axes", labelsize=18) # 18
plt.rc("xtick", labelsize=14, top=True, direction="in")
plt.rc("ytick", labelsize=14, right=True, direction="in")
plt.rc("axes", titlesize=18)
plt.rc("legend", fontsize=18)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.minor.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.major.size'] = 7.5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 7.5
plt.rcParams['ytick.minor.size'] = 3

''' SETUP '''
# Backgounds
p_back = get_average('data/Day2/p_background.csv')
s_back = get_average('data/Day2/s_background.csv')

# Intensities
Ip = get_average('data/Day2/int_p.csv')
Is = get_average('data/Day2/int_s.csv')

# Angles
phi_Ts = load_angles('data/Day2/p_angles.txt')
phi_Rs = phi_Ts[:,5:]
phi_Tp = load_angles('data/Day2/s_angles.txt')
phi_Rp = phi_Tp[:,3:]

# Da real intenities
Tp = load_intensities('Tp', phi_Tp[2,:], Ip, p_back)
Ts = load_intensities('Ts', phi_Ts[2,:], Is, s_back)
Rp = load_intensities('Rp', phi_Rp[2,:], Ip, p_back)
Rs = load_intensities('Rs', phi_Rs[2,:], Is, s_back)

''' Lad os se hvor flækket vores data er! '''
# Find n
find_n(phi_Ts[0], phi_Ts[1], to_plot=True)

# Intensitetsplot. Ikke pænt endnu, det gør jeg senere!
plot_intensities()
