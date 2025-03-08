import matplotlib.pyplot as plt 
import matplotlib.colors as mcol
import matplotlib.cm as cm
import numpy as np
import seaborn as sns 
sns.set_style('ticks')

import pickle

import matplotlib as mpl 

from scipy.ndimage import gaussian_filter

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def bin_spectra(wavelengths, depths, depths_errors, bins):
    """
    Function that bins input wavelengths and transit depths (or any other observable, like flux) to a given 
    resolution `R`. Useful for binning transit depths down to a target resolution on a transit spectrum.

    Parameters
    ----------

    wavelengths : np.array
        Array of wavelengths
    
    depths : np.array
        Array of depths at each wavelength.

    bins : np.array
        Target bins

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
    dderr = depths_errors[idx]

    # Prepare output arrays:
    wout, dout, derrout = np.zeros(len(bins)), np.zeros(len(bins)), np.zeros(len(bins))

    for i in range( len(bins) ):

        if i == 0:

            half_bin = ( bins[i+1] - bins[i] ) * 0.5

        else:

            half_bin = ( bins[i] - bins[i-1] ) * 0.5

            
        idx = np.where( (ww > bins[i] - half_bin) & (ww < bins[i] + half_bin) )[0]

        wout[i] = np.mean( ww[idx] )
        dout[i] = np.mean( dd[idx] )
        derrout[i] = np.sqrt( np.sum( dderr[idx]**2 ) ) / len(idx)
        
    return wout, dout, derrout

def transit_depth_to_sh(depths, data_depths, depths_error, offset1, offset2, rstar, mplanet, tplanet, mu = 2.3):
    """
    Assuming input depths in ppm (an array), this returns back the value in Rp/H.
    Here Rp is the planetary radius, and H is the scale-height.

    depths: model depths in ppm
    data_depths : data depths in ppm
    depths_error : error on the data depths
    offset1 : offset in ppms of nrs1
    offset2 : offset in ppms of nrs2
    rstar : stellar radii in radius of the sun
    mplanet : mass of the planet in masses of the earth
    tplanet : temperature of the planet in K
    mu : (optional) mmw of the atmosphere
    """

    rsun = 695700000. # m
    rearth = 6378000. # m
    mearth = 5.972*1e24 # kg
    kB = 1.4*1e-23 # J/kg
    mp = 2*1e-27 # kg
    G = 6.67430*1e-11 # m3/kg/s

    # Transform depth in ppm to rp/rs:
    rprs = np.zeros(len(data_depths))
    rprs1 = np.zeros(len(data_depths))
    rprs2 = np.zeros(len(data_depths))
    rprs_err = np.zeros(len(data_depths))
    for i in range(len(data_depths)):

        distribution = np.sqrt( np.random.normal(data_depths[i]*1e-6, depths_error[i]*1e-6, 1000) )
        distribution1 = np.sqrt( np.random.normal( (data_depths[i] + offset1 )*1e-6, depths_error[i]*1e-6, 1000) )
        distribution2 = np.sqrt( np.random.normal( (data_depths[i] + offset2 )*1e-6, depths_error[i]*1e-6, 1000) )
        rprs[i] = np.mean(distribution)
        rprs1[i] = np.mean(distribution1)
        rprs2[i] = np.mean(distribution2)
        rprs_err[i] = np.sqrt(np.var(distribution))

    # Next, get rp in meters:
    rp = rprs * rstar * rsun
    rp1 = rprs1 * rstar * rsun
    rp2 = rprs2 * rstar * rsun

    # Transform depth in ppm to rp:
    rp_out = np.sqrt(depths * 1e-6) * rstar * rsun

    # Now, get scale-height:
    print('Estimated planet radius: {0:.2f} Rearth'.format( np.nanmedian(rp) / rearth ) )
    g = ( G * mplanet * mearth ) / ( np.nanmedian(rp) )**2
    H = ( kB * tplanet ) / ( mu * mp * g )
    print('Estimated gravity: {0:.2f} m/s2'.format( g ) )
    print('Estimated scale-height: {0:.2f} km'.format( H/1e3 ) )

    # And return scaled rp/rs:
    return rp_out / H, rp1 / H, rp2 / H

# Plotting
plt.figure(figsize=(6, 7)) 

# Read-in pickle file with spectra:
spectra = pickle.load(open('sub-neptune-jwst-spectra.pkl','rb'))

# Read-in pickle file with models:
models = pickle.load(open('sub-neptune-jwst-model-spectra.pkl','rb'))

# Pop GJ 9827, because it doesn't have 3-5 um spectra:
spectra.pop('GJ 9827 d')

# Define bins to bin spectra:
bins1 = np.linspace(3,3.6,5)#8)
bins2 = np.linspace(3.8,5,10)#12)

# Extract temperatures and masses for all planets first:
planets = list(spectra.keys())
temperatures = np.zeros(len(planets))
masses = np.zeros(len(planets))
for i in range(len(planets)):

    temperatures[i] = spectra[planets[i]]['planet teq']
    masses[i] = spectra[planets[i]]['planet mass']

# Plot planets ordered by mass, color them by temperature. Set colormap:
#cm1 = sns.cubehelix_palette(as_cmap=True, start = 0, rot = 0.8, hue = 0.5, light = 0.85, dark = 0.2)#sns.color_palette("inferno", as_cmap=True)# mcol.LinearSegmentedColormap.from_list("MyCmapName",["orangered","cornflowerblue"])
#cm1 = sns.color_palette("rocket", as_cmap=True)
#cm1 = sns.color_palette(["#3B0F70", "#8C2981", "#DD4968", "#FE9F50", "#FCDEA2"], as_cmap=True)
#cm1 = sns.color_palette("RdBu", 10)
cm1 = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
cnorm = mcol.Normalize(vmin=min(temperatures),vmax=max(temperatures))
cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
cpick.set_array([])

# Sort by mass:
#idx = np.argsort(masses)
# Sort by temperature
idx = np.argsort(temperatures)

counter = 0
offset = 6.5
start = - offset * ( len(planets)/2. ) 
gfw = 3
for i in idx:

    # Plot model fits:
    binned_model, rp1, rp2 = transit_depth_to_sh(models[planets[i]]['binned best fit'],
                                       spectra[planets[i]]['depth'],
                                       spectra[planets[i]]['depth_errors'],
                                       models[planets[i]]['NRS1 offset'],
                                       models[planets[i]]['NRS2 offset'],
                                       spectra[planets[i]]['stellar radius'], 
                                       spectra[planets[i]]['planet mass'], 
                                       spectra[planets[i]]['planet teq'], mu = 2.3)

    model, _, _ = transit_depth_to_sh(models[planets[i]]['model best fit'], 
                                       spectra[planets[i]]['depth'],
                                       spectra[planets[i]]['depth_errors'],
                                       models[planets[i]]['NRS1 offset'],
                                       models[planets[i]]['NRS2 offset'],
                                       spectra[planets[i]]['stellar radius'], 
                                       spectra[planets[i]]['planet mass'], 
                                       spectra[planets[i]]['planet teq'], mu = 2.3)

    idx_nrs1 = np.where(spectra[planets[i]]['wavelength'] < 3.8)[0]
    idx_nrs2 = np.where(spectra[planets[i]]['wavelength'] > 3.8)[0]
    rp1 = rp1 - np.nanmedian(binned_model)
    rp2 = rp2 - np.nanmedian(binned_model)
    model = model - np.nanmedian(binned_model)
    binned_model = binned_model - np.nanmedian(binned_model)

    plt.plot( models[planets[i]]['model wavelength'], 
              model + start + counter*offset, 
              color = cpick.to_rgba(temperatures[i])
            )

    plt.errorbar( spectra[planets[i]]['wavelength'][idx_nrs1],
                  rp1[idx_nrs1] + start + counter*offset, 
                  spectra[planets[i]]['normalized_rp_errors'][idx_nrs1],
                  fmt = '.', ms = 3, elinewidth = 1, alpha = 0.3, color = cpick.to_rgba(temperatures[i]) 
                )

    plt.errorbar( spectra[planets[i]]['wavelength'][idx_nrs2],
                  rp2[idx_nrs2] + start + counter*offset,        
                  spectra[planets[i]]['normalized_rp_errors'][idx_nrs2],
                  fmt = '.', ms = 3, elinewidth = 1, alpha = 0.3, color = cpick.to_rgba(temperatures[i]) 
                )

    wb1, db1, dberr1 = bin_spectra( spectra[planets[i]]['wavelength'][idx_nrs1],
                                 rp1[idx_nrs1] + start + counter*offset,
                                 spectra[planets[i]]['normalized_rp_errors'][idx_nrs1], 
                                 bins1 
                               )

    wb2, db2, dberr2 = bin_spectra( spectra[planets[i]]['wavelength'][idx_nrs2],
                                 rp2[idx_nrs2] + start + counter*offset,
                                 spectra[planets[i]]['normalized_rp_errors'][idx_nrs2],
                                 bins2 
                               )

    plt.errorbar( wb1, 
                  db1,
                  dberr1,
                  fmt = 'o', ms = 5, elinewidth = 1, color = cpick.to_rgba(temperatures[i]), #mec = 'grey',#, mfc = 'white'
                )

    plt.errorbar( wb2,
                  db2,
                  dberr2,
                  fmt = 'o', ms = 5, elinewidth = 1, color = cpick.to_rgba(temperatures[i]), #mec = 'grey',#, mfc = 'white'
                )

    plt.text( 4.8, np.nanmedian( spectra[planets[i]]['normalized_rp'] + start + counter*offset ), planets[i], color = cpick.to_rgba(temperatures[i]))

    counter += 1

#plt.xscale('log')

# Set labels, fontsizes:
#plt.ylim(-10,10)
plt.xlim(2.9, 5.05)
plt.xticks([3., 4., 5.], ['3', '4', '5'], fontsize = 16) 
plt.yticks(fontsize = 16) 
plt.ylim(-22,13)
plt.ylabel('$R_p/H$ + offset', fontsize = 18)
plt.xlabel('Wavelength ($\mu$m)', fontsize = 18)
plt.colorbar(cpick,pad=0.1,label="$T_{eq}$", ax = plt.gca())
plt.tight_layout()
plt.savefig('espinoza_pre-figure8.pdf', dpi=300)
# Show the plot
plt.show()
