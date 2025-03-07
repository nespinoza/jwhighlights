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
plt.figure(figsize=(5, 4)) 

mw, mf = np.loadtxt('model.txt', unpack = True, usecols = (0, 1))
idx = np.argsort(mw)
mw, mf = mw[idx], mf[idx]

nf = np.max(mf)
plt.plot(mw, ( mf / nf ) + 0.04, 'r-')
w,wl,wr,f,ferr = np.loadtxt('data.txt', unpack = True, usecols = (0,1,2,3,4))
idx = np.argsort(w)
werr = (wr-wl)*0.5
plt.errorbar(w, ( f * 1e18 ) / nf, xerr = werr, yerr= ( ferr * 1e18 ) / nf, color = 'black', fmt = '.', ms = 5, elinewidth = 1)

# Set labels, fontsizes:
#plt.ylim(20260, 22880)
plt.xlim(10,16)
plt.xticks([10, 12, 14, 16], ['10', '12', '14', '16'], fontsize = 16)
plt.yticks(fontsize = 16) 
plt.ylabel('Relative flux', fontsize = 18)
plt.xlabel('Wavelength ($\mu$m)', fontsize = 18)
plt.tight_layout()
plt.savefig('espinoza_pre-figure4_malin.pdf', dpi=300)
# Show the plot
plt.show()
