#%% 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks

#%% 
def find_nearest(array,value): 
    vals = np.abs(array-value)
    return np.argmin(vals)

def load_data(file_name): 
    #file_path = 'E:\Eksperimentel_fysik\Spektroskopi\Spektroskopi_Data\\' + file_name + '.txt'
    file_path = 'data/He.txt'
    wavelength = []
    ints = []
    with open(file_path,'r') as file: 
        for line in file: 
            if line[0].isnumeric():
                dims = line.replace(',','.')
                dims = dims.split('\t')
                wavelength.append(float(dims[0]))
                ints.append(float(dims[1]))

    wavelength = np.array(wavelength)
    ints = np.array(ints)
    return wavelength,ints 

def read_NIST_ASCII(file_name): 
    #file_path = 'E:\Eksperimentel_fysik\Spektroskopi\Spektroskopi_Data\\' + file_name + '.txt'
    file_path = 'data/He_lines.txt'
    wavelength = []
    upper_level = []
    lower_level = []

    with open(file_path,'r') as file: 
        for line in file: 
            if line[:2] == 'He': 
                dims = line.replace(' ','')
                dims = dims.split('|')
                wavelength.append(float(dims[1]))
                lower_level.append(dims[7] + ' ' + dims[8])
                upper_level.append(dims[10] + ' ' + dims[11])
              
    lower_level = np.array(lower_level)
    upper_level = np.array(upper_level)
    unique_wave,unique_index = np.unique(wavelength,return_index=True)
    return unique_wave,lower_level[unique_index],upper_level[unique_index] 

def find_match(wave_data,wave_ref,lower_levels,upper_levels,uncertainty=2,to_print=True):  
    diffs = np.array([np.abs(i-wave_ref) for i in wave_data]) #Rækker er indgange i data¨
    close_match = diffs < uncertainty
    
    if to_print: 
        for data_i,row_filter in enumerate(close_match): 
            wave_match = wave_ref[row_filter]
            lower_match = lower_levels[row_filter]
            upper_match = upper_levels[row_filter]
            print(f'{wave_data[data_i]}:')
            for j,match in enumerate(wave_match): 
                print(f'    {match} : {lower_match[j]} - {upper_match[j]}')
    return close_match

def plot_top(wave_peaks,int_peaks,wave_ref,lower_levels,upper_levels,wavelength,ints,uncertainty=2,nr_tops=5):
    if nr_tops > 9: 
        print('YO makker maks 9 peaks!')
        return None 
    
    color_names = ['Orange','Green','Red','Purple','Brown','Pink','Gray','Olive','Cyan']
    c_list = [f'C{i+1}' for i in range(len(color_names))]
    
    #Find det data der skal bruges: 
    sort_inds = np.argsort(int_peaks)[::-1] 
    sort_inds = sort_inds[:nr_tops]
    wave_peaks = wave_peaks[sort_inds]
    int_peaks = int_peaks[sort_inds]

    #Plot og print! 
    fig,ax = plt.subplots()
    ax.plot(wavelength,ints)
    match_mat = find_match(wave_peaks,wave_ref,lower_levels,upper_levels,to_print=False)
    for data_i,row_filter in enumerate(match_mat): 
        wave_match = wave_ref[row_filter]
        lower_match = lower_levels[row_filter]
        upper_match = upper_levels[row_filter]
        print(f'{color_names[data_i]} - {wave_peaks[data_i]}:')
        for j,match in enumerate(wave_match): 
                print(f'    {match} : {lower_match[j]} - {upper_match[j]}')
        ax.plot(wave_peaks[data_i],int_peaks[data_i],'o',c=c_list[data_i])
    plt.show()
#%% 
wavelength,ints = load_data('He')
lower_index = find_nearest(wavelength,300)
upper_index = find_nearest(wavelength,800)

wavelength = wavelength[lower_index:upper_index+1]
ints = ints[lower_index:upper_index+1]

peaks = find_peaks(ints,height=1000)[0]
wave_peaks = wavelength[peaks]
int_peaks = ints[peaks]

plt.plot(wavelength,ints)
plt.plot(wavelength[peaks],ints[peaks],'o')
#%% 
wave_ref,lower_levels,upper_levels = read_NIST_ASCII('He_lines')
#match_mat = find_match(wave_peaks,wave_ref,lower_levels,upper_levels,to_print=True)
plot_top(wave_peaks,int_peaks,wave_ref,lower_levels,upper_levels,wavelength,ints,nr_tops=5)
