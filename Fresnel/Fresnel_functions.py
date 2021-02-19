import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.stats import chi2

def get_average(file):
    data = np.loadtxt(file,skiprows=2,usecols=1,delimiter=',')
    mean = np.mean(data)
    std = np.std(data,ddof=1)
    error = std/np.sqrt(len(data))
    return mean,error


def load_angles(angle_file):
    angle_list = np.loadtxt(angle_file,skiprows=2,delimiter=',').T
    phi1 = angle_list[0] #Vinkel af glasset
    phi2 = phi1 - angle_list[1] #Vinkel af transmission
    return np.array([np.deg2rad(phi1),np.deg2rad(phi2),phi1])


def find_n(phi1,phi2,to_plot=False):
    def fit_func(sin1,a,b):
        return a*sin1+b
    phi2_err = np.ones_like(phi2)*np.pi/180*np.sqrt(2)*0.5 #Fejlpropagering gennem deg -> rad
    sin1 = np.sin(phi1)
    sin2 = np.sin(phi2)
    sin2_err = np.abs(np.cos(phi2))*phi2_err
    p_opt, p_cov = curve_fit(fit_func, sin1, sin2, sigma=sin2_err, absolute_sigma=True)
    p_err = np.sqrt(np.diag(p_cov))
    ch_min = np.sum(((sin2-fit_func(sin1, p_opt[0],p_opt[1]))/sin2_err)**2)
    p_val = chi2.cdf(ch_min,len(sin2)-2)
    n = 1/p_opt[0] #Antager midlertidigt at n_luft = 1
    n_err = p_err[0]/p_opt[0]**2

    if to_plot:
        plt.errorbar(sin1,sin2,yerr=sin2_err,fmt='o', ms=5, lw=2, capsize=5)
        plt.plot(sin1,fit_func(sin1,p_opt[0],p_opt[1]))
        plt.show()
    return (n,n_err,p_val)


def load_intensities(file_type,angles,int0,background): #OBS! Angles must be deg! Use load_angles[3].
    #Filetype = 'Tp'/'Ts' osv.
    mean_list = []
    error_list = []
    mean_background, error_background = background
    mean_int, error_int = int0
    mean_int = mean_int - background #Korriger for baggrund
    error_int = np.sqrt(error_int**2+error_background**2) #Error prop på usikkerheden

    for angle in angles:
        file = 'data/Day2/' + file_type + f'{int(angle)}.csv'
        mean,error = get_average(file)
        mean = mean - background #Subtracts background
        error = np.sqrt(error**2+error_background**2) #Error prop with background
        rel_mean = mean/mean_int
        error_prop = rel_mean * np.sqrt((error/mean)**2+(error_int/mean_int)**2)
        mean_list.append(rel_mean)
        error_list.append(error_prop)
    return np.array([mean_list,error_list])


def plot_intensities():
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8,8))
    xs = np.linspace(0, np.pi, 1000)

    # Tp
    ax[0,0].errorbar(angles, Tp[0,:], yerr=Tp[1,:], fmt='o')
    ax[0,0].plot(xs, Tp_model(xs))

    # Ts
    ax[0,1].errorbar(angles, Ts[0,:], yerr=Ts[1,:], fmt='o')
    ax[0,1].plot(xs, Ts_model(xs))

    # Rp
    ax[1,0].errorbar(angles, Rp[0,:], yerr=Rp[1,:], fmt='o')
    ax[1,0].plot(xs, Rp_model(xs))

    # Rs
    ax[1,1].errorbar(angles, Rs[0,:], yerr=Rs[1,:], fmt='o')
    ax[1,1].plot(xs, Rs_model(xs))


def theta2(theta1):
    return np.arcsin((1/1.5)*np.sin(theta1))

def Rp_model(theta1):
    return (np.tan(theta1 - theta2(theta1)))**2/(np.tan(theta1 + theta2(theta1)))**2

def Rs_model(theta1):
    return (np.sin(theta1 - theta2(theta1)))**2/(np.sin(theta1 + theta2(theta1)))**2

def Tp_model(theta1):
    return np.sin(2*theta1)*np.sin(2*theta2(theta1))/((np.sin(theta1 + theta2(theta1)))**2*(np.cos(theta1 - theta2(theta1))**2))*(1.5/1*4/(1.5+1)**2)

def Ts_model(theta1):
    return np.sin(2*theta1)*np.sin(2*theta2(theta1))/(np.sin(theta1 + theta2(theta1)))**2*(1.5*4/(1.5+1)**2)
