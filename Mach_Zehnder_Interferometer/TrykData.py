#%%
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.stats import chi2

#%% 
# Preamble:
plt.rc('text', usetex=True)
plt.rc("axes", labelsize=18)
plt.rc("xtick", labelsize=18, top=True, direction="in")
plt.rc("ytick", labelsize=18, right=True, direction="in")
plt.rc("axes", titlesize=18)
plt.rc("legend", fontsize=15)
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.minor.size'] = 3

#%%
p = np.array([1, 0.5, 0.75,0.25, 0.6, 0.8])
m = [[46,47,46,44,46],[24,23,21,20,21,22,22,21],[35,34,32,30,31,34,36,34,34],[11,11,12,12,11],[28,26,25,26,27],[35,36,35,35,37]]

l = 0.0587 #m
lambd = 633e-9 #m
l_err = 0.00005 #m 


m_error = [np.std(i,ddof=1)/np.sqrt(len(i)) for i in m] #Skal have fejl for tælletal? 
m = np.array([np.mean(i) for i in m])
p_error = np.array([0.05 if i!=0.25 else 0.125/2 for i in p])

#Lav fit til skidtet! 
def fit_func(x,a,b): 
    return a*x + b

def inv_fit(x,a,b): 
    return (x-b)/a

p_opt, p_cov = curve_fit(fit_func,m,p,sigma=p_error,absolute_sigma=True)
p_err = np.sqrt(np.diag(p_cov))

ch_min = np.sum(((p-fit_func(m, p_opt[0],p_opt[1]))/p_error)**2)
p_val = chi2.cdf(ch_min,4)
print(f'p_val : {p_val}')


l_param = l*p_opt[0]
l_param_err = l_param*np.sqrt((p_err[0]/p_opt[0])**2+(l_err/l)**2)

a = lambd/(2*l_param)
a_err = lambd*l_param_err/(2*l_param**2)


atm =1.01325
atm_err = 0.00005
n0 = a*atm+1
n0_err = (n0-1)*np.sqrt((a_err/a)**2+(atm_err/atm)**2) 

print(f'a : {a} \u00b1 {a_err}')
print(f'n0 : {n0} \u00b1 {n0_err}')

plt.rcParams.update({'figure.autolayout': True})
plt.errorbar(p,m,yerr=m_error,xerr=p_error,fmt='o',ms=5,lw=2,capsize=3)
plt.plot(p,inv_fit(p,p_opt[0],p_opt[1]))
plt.xlabel('Pressure difference (bar)')
plt.ylabel('Number of peaks')
plt.savefig('pressure_fit.eps',format='eps')

#plt.errorbar(m,p,yerr=p_error,xerr=m_error,fmt='o',ms=5,lw=2,capsize=3)
#plt.plot(m,fit_func(m,p_opt[0],p_opt[1]))


# %%
