#%% 
import numpy as np 
import matplotlib.pyplot as plt 


#%% 
def find_nearest(array,value): 
    vals = np.abs(array-value)
    return np.argmin(vals)

#%% 
name = 'Ne'
file_path = 'E:\Eksperimentel_fysik\Spektroskopi\Spektroskopi_Data\\' + name + '.txt'
#data = np.loadtxt('E:\Eksperimentel_fysik\Spektroskopi\Spektroskopi_Data\\' + name + '.txt',header='ASCII',skiprows=15).T

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

index_500 = find_nearest(wavelength,500)
index_800 = find_nearest(wavelength,800)

plt.plot(wavelength[index_500:index_800],ints[index_500:index_800])
#%% 
test = 'Hej med dig'
test.split()