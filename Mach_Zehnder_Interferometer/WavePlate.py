#%%
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


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
def load_file(name): 
    #data = np.loadtxt('data/' + name + '.csv',skiprows=2,delimiter=',').T 
    data = np.loadtxt('E:\Eksperimentel_fysik\Mach_Zehnder\Dag15\\' + name + '.csv',skiprows=2,delimiter=',').T
    return data

def find_slopes(top, bot):
    rise = []
    fall = []
    first_rising = top[0] > bot[0]          
    last_rising = top[-1] > bot[-1]
    max_len = max(len(top), len(bot))
    if first_rising and last_rising:
        fall = [(top[i], bot[i+1]) for i in range(max_len - 1)]
        rise = [(bot[i], top[i] ) for i in range(max_len)]
    elif first_rising and not last_rising:
        fall = [(top[i], bot[i+1]) for i in range(max_len - 1)]
        rise = [(bot[i],top[i]) for i in range(max_len - 1)]
    elif not first_rising and not last_rising:
        fall = [(top[i], bot[i]) for i in range(max_len)]
        rise = [(bot[i],top[i+1]) for i in range(max_len - 1)]
    elif not first_rising and last_rising:
        fall = [(top[i], bot[i]) for i in range(max_len - 1)]
        rise = [(bot[i],top[i+1]) for i in range(max_len -1)]
    else:
        print('You fucked it up fam')
    return rise, fall

def slope_intensity(slope,volt,intensity): 
    volt_slope = volt[slope[0]:slope[1]+1]
    intensity_slope = intensity[slope[0]:slope[1]+1]
    unique_volt = sorted(list(set(volt_slope)))
    slope_int = [np.mean([intensity_slope[i] for i,volt_j in enumerate(volt_slope) if volt_j == volt_i]) for volt_i in unique_volt]
    return slope_int, unique_volt

def mean_several_files(max_file_num,file_name,N,to_print=False): 
    rise_ints = []
    volt_to_return = []
    for i in range(max_file_num): 
        print(i)
        
        if i < 9: 
            _,intensity,volt = load_file(f'{file_name}_0{i+1}')
        else: 
            _,intensity,volt = load_file(f'{file_name}_{i+1}')
        if len(volt) < 1000:
            continue 
        else: 
            max_peaks = find_peaks(volt,distance=1000,width=100)[0]
            min_peaks = find_peaks(-1*volt,distance=1000,width=100)[0]
            rise_slopes,fall_slopes = find_slopes(max_peaks,min_peaks)

            for rise in rise_slopes:
                s_int,unique_volt = slope_intensity(rise,volt,intensity)
                if len(unique_volt) == N:
                    rise_ints.append(s_int)
                    volt_to_return = unique_volt
                if to_print: 
                    print(f'Rise:{len(unique_volt)}')
                
    rise_mean = np.mean(rise_ints,axis=0)
    rise_error = np.std(rise_ints,ddof=1,axis=0)/np.sqrt(len(rise_ints))
    return volt_to_return,rise_mean,rise_error

def find_N(max_file_num,file_name): 
    def most_frequent(List):
        return max(set(List), key = List.count)
    
    N_list = []
    
    for i in range(max_file_num): 
        #print(i)
        
        if i < 9: 
            _,intensity,volt = load_file(f'{file_name}_0{i+1}')
        else: 
            _,intensity,volt = load_file(f'{file_name}_{i+1}')
        if len(volt) < 1000:
            continue 
        else: 
            max_peaks = find_peaks(volt,distance=1000,width=100)[0]
            min_peaks = find_peaks(-1*volt,distance=1000,width=100)[0]
            rise_slopes,fall_slopes = find_slopes(max_peaks,min_peaks)

            for rise in rise_slopes:
                s_int,unique_volt = slope_intensity(rise,volt,intensity)
                N_list.append(len(unique_volt))
    return most_frequent(N_list)

def Gauss_fit(x,y,y_err,guess,to_plot=False): 
    def Gauss(x,a,b,c): 
        return a*np.exp(-(x-b)**2/(2*c**2))
    p_opt, p_cov = curve_fit(Gauss,x,y,p0=guess, sigma=y_err, absolute_sigma=True)
    p_err = np.sqrt(np.diag(p_cov))
    
    if to_plot: 
        plt.plot(x,y)
        plt.plot(x,Gauss(x,p_opt[0],p_opt[1],p_opt[2]))
    return p_opt[0],p_err[0] #Retunerer middelværdien! 

def find_single_max(x,y,y_err,peak_nr=1,width=5,to_plot=False): 
    peaks = find_peaks(y,width=2)[0]
    if len(peaks) < 2: 
        mean = np.mean(y)
        mean_err = np.std(y,ddof=1)/np.sqrt(len(y))
        return mean,mean_err
    peak = peaks[peak_nr]

    y_fit = y[peak-width:peak+width+1]
    x_fit = x[peak-width:peak+width+1]
    y_fit_err = y_err[peak-width:peak+width+1]
    guess = [x[peak],5,5]
    mean,mean_err = Gauss_fit(x_fit,y_fit,y_fit_err,guess,to_plot=to_plot)
    return mean,mean_err        

def find_max(x,y,y_err,width=5,to_plot=False): 
    max_mean_list = []
    max_error_list = []
    min_mean_list = []
    min_error_list = []

    max_min = [1,-1]

    for extreme_type in max_min: 
        if extreme_type == -1: 
            min_val = min(-1*y)
            y = -1*y - min_val
        
        peaks = find_peaks(y,width=2)[0]
        if len(peaks) == 0: 
            min_mean_list.append(max(y))
            continue

        for peak in peaks: 
            if peak+1 < 6 or len(x)-peak+1 < 6: 
                width = min(peak,len(x)-peak+1)
            y_fit = y[peak-width:peak+width+1]
            x_fit = x[peak-width:peak+width+1]
            y_fit_err = y_err[peak-width:peak+width+1]
            guess = [y[peak],x[peak],4]

            #plt.plot(x_fit,y_fit)
            #plt.show()

            mean,mean_err = Gauss_fit(x_fit,y_fit,y_fit_err,guess,to_plot=to_plot)

            if extreme_type == 1: 
                max_mean_list.append(mean)
                max_error_list.append(mean_err)
            else: 
                min_mean_list.append(mean)
                min_error_list.append(mean_err)
        
    top_mean = np.mean(max_mean_list)
    bottom_mean = -1*(np.mean(min_mean_list) + min_val)
    top_bot_mean = np.mean([top_mean,bottom_mean])

    ###Error prop - holder kun for to peaks!###
    if len(max_error_list) <2: 
        final_error = 'NoOoO!'
        print('Hej!')
    else: 
        print(max_error_list)
        top_mean_error = np.sqrt(max_error_list[0]**2 + max_error_list[1]**2)/2 #/2 fordi middelværdi
        bottom_mean_error = np.sqrt(min_error_list[0]**2 + min_error_list[1]**2)/2 #/2 fordi middelværdi
        top_bot_mean_error = np.sqrt(top_mean_error**2 + bottom_mean_error**2)/2 
        final_error = np.sqrt(top_mean_error**2 + top_bot_mean_error**2)
    return (top_mean-top_bot_mean,final_error,(top_mean,bottom_mean))

#%% 
angles = [19,22,25,30,35,40,45,50,55,60,65,67,70,75,80,85,90,100,105,110,115,120]
N_list = []

print('Begynder på:')
for angle in angles: 
    print(angle)
    N_list.append(find_N(5,f'Int_{angle}/Int_{angle}'))
print(N_list)

#%% 
N_nr = 0
volts, mean, error = mean_several_files(3,f'Int_{angles[N_nr]}/Int_{angles[N_nr]}',N=N_list[N_nr],to_print=False)
top,top_err,maxmin_mean,_ = find_max(volts,mean,error,to_plot=False)#peak_nr=0)
plt.plot(volts,mean)
plt.plot(volts,[maxmin_mean[0]]*len(volts))
plt.plot(volts,[maxmin_mean[1]]*len(volts))
plt.show()


#%% 
top_list = []
top_err_list = []
for i,angle in enumerate(angles):  
    print(f'{angle}:')
    volts, mean, error = mean_several_files(3,f'Int_{angle}/Int_{angle}',N=N_list[i],to_print=False)
    top,top_err,maxmin_mean = find_max(volts,mean,error,to_plot=False)
    plt.plot(volts,mean)
    plt.plot(volts,[maxmin_mean[0]]*len(volts))
    plt.plot(volts,[maxmin_mean[1]]*len(volts))
    plt.show()
    top_list.append(top)
    top_err_list.append(top_err)

#%% 
int_list = np.array(top_list)
int_err = np.array(top_err_list)

to_flip = np.argwhere(np.array(angles) == 70)[0,0] #Flip alle dem efter skiftet! 
to_die_index = to_flip-1 #Målingen i midten er underlig - Skyldes den måde vi behandler den på

int_list[to_flip:] = -1*int_list[to_flip:]# - min(int_list[to_flip:])
int_list = np.delete(int_list,to_die_index)
int_err = np.delete(int_err,to_die_index)
int_err = int_err.astype('float')

min_index = np.argmin(int_list)
max_index = np.argmax(int_list)



int_list = int_list + -1*int_list[min_index] #Sæt min int til at være 0 
int_err = np.sqrt(int_err**2+int_err[min_index]**2) #Tilhørende fejlprop. 
normed_int_list = int_list.copy()/int_list[max_index] #Normer 
int_err = normed_int_list * np.sqrt((int_err/int_list)**2+(int_err[max_index]/int_list[max_index])**2)
int_err = np.nan_to_num(int_err,nan=0.001)

def cos_fit(theta,A,omega,phi): 
    return (A*np.cos(omega*theta+phi))**2 

rads = np.deg2rad(angles)
rads = np.delete(rads,to_die_index)
rads = rads - min(rads)
rads_err = np.deg2rad([1.0]*len(rads))

p_opt, p_cov = curve_fit(cos_fit,rads,normed_int_list,p0=[1,1,0])#,sigma=int_err,absolute_sigma=True)
print(p_opt)
print(np.sqrt(np.diag(p_cov)))
plt.rcParams.update({'figure.autolayout': True})
plt.errorbar(rads,normed_int_list,xerr=rads_err,yerr=int_err,fmt='o',ms=5,lw=2,capsize=3)
plt.plot(rads,cos_fit(rads,p_opt[0],p_opt[1],p_opt[2]))
plt.xlabel(r'$\theta$ (rad)')
plt.ylabel('Relative intensity')
plt.savefig('Cos2.eps',format='eps')
#Hæv alle punkter med middelværdien af 67 (det er vores 'nulpunkt'!)

#%% 
test = np.array([1,2,3])
test = np.delete(test,1)
print(test)

if [1,2,3]: 
    print('Hej')