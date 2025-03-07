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

def bin_at_resolution(wavelengths, depths, depths_errors = None, R = 100, method = 'mean'):
    """
    Function that bins input wavelengths and transit depths (or any other observable, like flux) to a given 
    resolution `R`. Useful for binning transit depths down to a target resolution on a transit spectrum.

    Parameters
    ----------

    wavelengths : np.array
        Array of wavelengths
    
    depths : np.array
        Array of depths at each wavelength.

    R : int
        Target resolution at which to bin (default is 100)

    method : string
        'mean' will calculate resolution via the mean --- 'median' via the median resolution of all points 
        in a bin.

    Returns
    -------

    wout : np.array
        Wavelength of the given bin at resolution R.

    dout : np.array
        Depth of the bin.

    derrout : np.array
        Error on depth of the bin.
    

    """

    # Sort wavelengths from lowest to highest:
    idx = np.argsort(wavelengths)

    ww = wavelengths[idx]
    dd = depths[idx]

    if depths_errors is not None:

        dderr = depths_errors[idx]

    # Prepare output arrays:
    wout, dout, derrout = np.array([]), np.array([]), np.array([])

    oncall = False

    # Loop over all (ordered) wavelengths:
    for i in range(len(ww)):

        if not oncall:

            # If we are in a given bin, initialize it:
            current_wavs = np.array([ww[i]])
            current_depths = np.array(dd[i])
            
            if depths_errors is not None:

                current_depth_errors = np.array(dderr[i])

            oncall = True

        else:

            # On a given bin, append next wavelength/depth:
            current_wavs = np.append(current_wavs, ww[i])
            current_depths = np.append(current_depths, dd[i])
    
            if depths_errors is not None:

                current_depth_errors = np.append(current_depth_errors, dderr[i])

            # Calculate current mean R:
            if method == 'mean':

                current_R = np.mean(current_wavs) / np.abs(current_wavs[0] - current_wavs[-1])

            elif method == 'median':

                current_R = np.median(current_wavs) / np.abs(current_wavs[0] - current_wavs[-1])

            else:

                raise Exception('Method '+method+' not supported. Try "mean" or "median".' )

                

            # If the current set of wavs/depths is below or at the target resolution, stop, save 
            # and move to next bin:
            if current_R <= R:

                wout = np.append(wout, np.mean(current_wavs))
                dout = np.append(dout, np.mean(current_depths))

                if depths_errors is None:

                    derrout = np.append(derrout, np.sqrt(np.var(current_depths)) / np.sqrt(len(current_depths)))

                else:

                    errors = np.sqrt( np.sum( current_depth_errors**2 ) ) / len(current_depth_errors)
                    derrout = np.append(derrout, errors )

                oncall = False

    return wout, dout, derrout



f = h5py.File('Eureka!-wasp-39b-spectrum.h5')

wavelength = f['wavelength'][:]
depth = f['dppm'][:]
depth_error = f['dppm_error'][:]

f.close()

plt.errorbar(wavelength[:-8], depth[:-8], depth_error[:-8], fmt = '.', color = 'black', ms = 3, elinewidth = 1)

all_ws = np.array([])
all_depths = np.array([])
all_depths_errors = np.array([])

for instrument in ['NIRCam_F322W2.csv', 'NIRISS_Order1.csv', 'NIRISS_Order2.csv', 'NIRSpec_G395H_NRS1.csv', 'NIRSpec_G395H_NRS2.csv']:#, 'NIRSpec_PRISM.csv']:

    w,rp,rperr1,rperr2 = np.loadtxt(instrument, unpack=True, skiprows=1, usecols = (1,4,5,6), delimiter = ',')
    rperr = (rperr1 + rperr2) * 0.5
    d,derr = np.zeros(len(w)), np.zeros(len(w))

    for i in range(len(w)):

        d_samples = ( np.random.normal(rp[i],rperr[i],300)**2 ) * 1e6
        d[i], derr[i] = np.mean(d_samples), np.sqrt(np.var(d_samples))

    #plt.errorbar(w, d, derr, fmt = '.')

    all_ws = np.append(all_ws, w)
    all_depths = np.append(all_depths, d)
    all_depths_errors = np.append(all_depths_errors, derr)

idx = np.argsort(all_ws)
wb, db, dberr = bin_at_resolution(all_ws[idx], all_depths[idx], depths_errors = all_depths_errors[idx], R = 150)

#plt.errorbar(wb, db, dberr, fmt = '.', color = 'black')

w = np.append(wb, wavelength[:-8])
d = np.append(db, depth[:-8])

idx = np.argsort(w)
gf = gaussian_filter(d[idx],1)

plt.plot(w[idx], gf, 'r', lw = 2)

idx = np.argsort(all_ws)
wb, db, dberr = bin_at_resolution(all_ws[idx], all_depths[idx], depths_errors = all_depths_errors[idx], R = 100)

plt.errorbar(wb, db, dberr, fmt = '.', color = 'black', ms = 3, elinewidth = 1)

plt.xscale('log')

# Set labels, fontsizes:
plt.ylim(20260, 22880)
plt.xlim(0.65,10.5)
plt.xticks([0.7, 1., 2., 3., 4., 5., 10.], ['0.7', '1', '2', '3', '4', '5', '10'], fontsize = 16) 
plt.yticks(fontsize = 16) 
plt.ylabel('Transit depth (ppm)', fontsize = 18)
plt.xlabel('Wavelength ($\mu$m)', fontsize = 18)
plt.tight_layout()
plt.savefig('espinoza_pre-figure3_w39.pdf', dpi=300)
# Show the plot
plt.show()


plt.show()
