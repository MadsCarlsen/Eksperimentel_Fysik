#%% 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks

#%% 
def find_nearest(array,value): 
    vals = np.abs(array-value)
    return np.argmin(vals)

def load_data(file_name,file_nr): 
    #file_path = 'E:\Eksperimentel_fysik\Spektroskopi\Spektroskopi_Data\\' + file_name + '.txt'
    file_path = f'data/{file_name}{file_nr}.txt'
    wavelength = []
    abs_val = []
    with open(file_path,'r') as file: 
        for line in file: 
            if line[0].isnumeric():
                dims = line.replace(',','.')
                dims = dims.split('\t')
                wavelength.append(float(dims[0]))
                abs_val.append(float(dims[1]))

    wavelength = np.array(wavelength)
    abs_val = np.array(abs_val)
    return wavelength,abs_val

#%%  
wavelength,abs_val = load_data('B',4)
lower_index = find_nearest(wavelength,200)
upper_index = find_nearest(wavelength,1200)

wavelength = wavelength[lower_index:upper_index+1]
abs_val = abs_val[lower_index:upper_index+1]

plt.plot(wavelength,abs_val)
plt.axvline()

#%% 
def get_abs(file_name,max_nr,target_wave): 
    abs_list = []
    for i in range(1,max_nr+1): 
        wave_data,abs_vals = load_data(file_name,i)
        wave_index = find_nearest(wave_data,target_wave)
        abs_list.append(abs_vals[wave_index])
    return np.array(abs_list)

abs_list = get_abs('B',4,650) 
c_list = [i / abs_list[-1] for i in abs_list]
plt.plot(c_list,abs_list,'o')
plt.xlabel('Relative concentration')
plt.ylabel('Absorbance')

