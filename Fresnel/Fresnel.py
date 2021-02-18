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
