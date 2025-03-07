import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
sns.set_style('ticks')

import h5py

import matplotlib as mpl 

from scipy.ndimage import gaussian_filter

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# Plotting
plt.figure(figsize=(6, 4)) 

f = h5py.File('WASP43b_MIRI_Data/2_Planetary_Spectra/eureka_v1.h5')
w = f['wavelength'][:]
fpfs = f['fp_fs'][:]
fpfs_errup = f['fp_fs_errorPos'][:]
fpfs_errdown = f['fp_fs_errorNeg'][:]
phases = f['phase']

for i in range( len(phases) ):

    #plt.plot(w,fpfs[i,:],'black',alpha = 0.1)
    plt.errorbar( w, fpfs[i, :]/1e3, yerr = [fpfs_errdown[i, :]/1e3, fpfs_errup[i, :]/1e3], fmt = 'o', color = 'black', mfc = 'white', ms = 7, elinewidth = 1 )

# Plot models. First, nightside:
for p in ['0.00','0.25','0.5','0.75']:

    w,m = np.loadtxt('models_phase'+p+'.txt', unpack = True, usecols = (0,1))
    idx = np.argsort(w)
    w, m = w[idx], m[idx]
    plt.plot(w,m/1e3, color = 'cornflowerblue')

# Set labels, fontsizes:
plt.xlim(5, 10.5)
plt.xticks([5, 6, 7, 8, 9, 10], ['5', '6', '7', '8', '9', '10'], fontsize = 16)
plt.yticks(fontsize = 16) 
plt.ylabel('$F_p/F_s$ (ppt)', fontsize = 18)
plt.xlabel('Wavelength ($\mu$m)', fontsize = 18)
plt.tight_layout()
plt.savefig('espinoza_pre-figure5_bell.pdf', dpi=300)
# Show the plot
plt.show()
