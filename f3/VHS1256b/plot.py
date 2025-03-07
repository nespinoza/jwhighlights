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
plt.figure(figsize=(10, 4)) 

w,f,ferr = np.loadtxt('VHS1256b_V2_accepted.txt', unpack = True, usecols = (0,1,2), delimiter = ',')
idx = np.argsort(w)
plt.plot(w[idx],f[idx]*1e16,color = 'black')

plt.xscale('log')

# Set labels, fontsizes:
#plt.ylim(20260, 22880)
plt.xlim(0.65,10.5)
plt.xticks([0.7, 1., 2., 3., 4., 5., 10.], ['0.7', '1', '2', '3', '4', '5', '10'], fontsize = 16) 
plt.yticks(fontsize = 16) 
plt.ylabel('Flux ($10^{-16}$ W/m$^2$/$\mu$m)', fontsize = 18)
plt.xlabel('Wavelength ($\mu$m)', fontsize = 18)
plt.tight_layout()
plt.savefig('espinoza_pre-figure3_v1256b.pdf', dpi=300)
# Show the plot
plt.show()
