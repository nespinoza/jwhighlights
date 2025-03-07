import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('ticks')

import matplotlib as mpl
from matplotlib.image import imread
import matplotlib.transforms as transforms

from utils import read_file

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

cycle = 'all'#'Cycle3'

# Turn on and off size of hexagons matching hours on target:
hours_on_target = False

# Turn on off naming of top targets:
names_hours_on_target = False#True

# Retrieve data for all exoplanets:
all_exoplanets = read_file('documents/PSCompPars_2025.02.05_16.15.06.csv')

# Retrieve properties for planets with measured masses and orbital distances:
idx = np.where( (all_exoplanets['pl_bmasse'] != '') & (all_exoplanets['pl_orbsmax'] != '') )[0]
all_exoplanet_masses = all_exoplanets['pl_bmasse'][idx].astype('float')
all_exoplanet_distances = all_exoplanets['pl_orbsmax'][idx].astype('float')

# Now retrieve properties for exoplanets observed by JWST:
jwst_exoplanets = read_file('documents/'+cycle+'.csv')

# Compile name of unique planets:
jwst_exoplanet_names = np.array(list(set(jwst_exoplanets['Planet'])))

# Prepare arrays:
jwst_direct_imaging = np.ones(len(jwst_exoplanet_names))
jwst_exoplanet_masses = np.zeros(len(jwst_exoplanet_names))
jwst_exoplanet_distances = np.zeros(len(jwst_exoplanet_names))
jwst_exoplanet_hours = np.zeros(len(jwst_exoplanet_names))

# Get properties for these:
for i in range(len(jwst_exoplanet_names)):

    jwst_exoplanet = jwst_exoplanet_names[i]
    # Compile time spent on this exoplanet:
    idx = np.where( jwst_exoplanet == jwst_exoplanets['Planet'] )
    jwst_exoplanet_hours[i] = np.sum(jwst_exoplanets['Seconds on target'][idx]) / 3600.
    
    for j in range(len(jwst_exoplanets['Planet'])):

        if jwst_exoplanet == jwst_exoplanets['Planet'][j]:

            jwst_exoplanet_masses[i] = jwst_exoplanets['Planet Mass (Earth masses)'][j]
            jwst_exoplanet_distances[i] = jwst_exoplanets['Planet semi-major axis (AU)'][j]

            if 'Direct Imaging' not in jwst_exoplanets['Science Mode'][j]:

                jwst_direct_imaging[i] = 0.

            break

# Plotting
fig = plt.figure(figsize=(10, 6))

# Define scale at the very beggining:
plt.xscale('log')
plt.yscale('log')

# Plot all the exoplanets:
plt.plot(all_exoplanet_distances, all_exoplanet_masses, '.', color = 'gainsboro', zorder = 1, rasterized=True)

# Plot JWST exoplanets. Identify direct imaging exoplanets:
idx_direct_imaging = np.where(jwst_direct_imaging == 1.)[0]
idx_non_direct_imaging = np.where(jwst_direct_imaging == 0.)[0]

# Plot all planets:
#plt.plot(jwst_exoplanet_distances, jwst_exoplanet_masses, 'h', mfc = '#F2CC8F', mec = 'black', ms = 10)

ymax, ymin = 300, 80 # maximum and minimum size 
if hours_on_target:

    # Determine sizes of markers:
    xmax, xmin = np.max(jwst_exoplanet_hours), np.min(jwst_exoplanet_hours)
    a = ( ymax - ymin ) / ( xmax - xmin )
    b = ymax - a * xmax
    sizes = a * jwst_exoplanet_hours + b

    print(sizes[idx_direct_imaging])
    # Plot directly imaged ones:
    plt.scatter(jwst_exoplanet_distances[idx_direct_imaging], 
             jwst_exoplanet_masses[idx_direct_imaging], 
             marker = 'h',
             color = ['#F2CC8F']*len(idx_direct_imaging), 
             edgecolor = ['black']*len(idx_direct_imaging), 
             s = sizes[idx_direct_imaging], zorder = 3) 

    # Plot non-directly imaged ones:
    plt.scatter(jwst_exoplanet_distances[idx_non_direct_imaging], 
             jwst_exoplanet_masses[idx_non_direct_imaging], 
             marker = 'h',
             color = ['#C4C6E7']*len(idx_non_direct_imaging), 
             edgecolor = ['black']*len(idx_non_direct_imaging), 
             s = sizes[idx_non_direct_imaging], zorder = 3)

    # Plot a special color for PSR J2322-2650, which is neither transiting not "directly imaged"
    idx = np.where( np.array(jwst_exoplanet_names) == 'PSR J2322-2650 b' )[0]
    print([jwst_exoplanet_distances[idx][0]])

    plt.scatter(jwst_exoplanet_distances[idx][0],
                jwst_exoplanet_masses[idx][0],
                marker = 'h', color = '#156064', edgecolor = 'black', s = float(sizes[idx][0]), zorder = 3) 

else:

    ##F2CC8F', '#C4C6E7
    # Plot directly imaged ones:
    plt.scatter(jwst_exoplanet_distances[idx_direct_imaging], 
             jwst_exoplanet_masses[idx_direct_imaging], 
             marker = 'h', color = '#F2CC8F', edgecolor = 'black', s = ymin)

    # Plot non-directly imaged ones:
    plt.scatter(jwst_exoplanet_distances[idx_non_direct_imaging], 
             jwst_exoplanet_masses[idx_non_direct_imaging], 
             marker = 'h', color = '#C4C6E7', edgecolor = 'black', s = ymin) 

    # Plot a special color for PSR J2322-2650, which is neither transiting not "directly imaged"
    idx = np.where( np.array(jwst_exoplanet_names) == 'PSR J2322-2650 b' )[0]
    plt.scatter([jwst_exoplanet_distances[idx]],
             [jwst_exoplanet_masses[idx]],
             marker = 'h', color = '#156064', edgecolor = 'black', s = ymin)

# Plot names of 10 top exoplanets in terms of hours-on-target:
if names_hours_on_target:

    n = 10
    idx = np.argsort(jwst_exoplanet_hours)[::-1][:n]
    print(jwst_exoplanet_distances[idx])
    print(jwst_exoplanet_masses[idx])
    print(jwst_exoplanet_names[idx])

    for i in idx:

        plt.text(jwst_exoplanet_distances[i], jwst_exoplanet_masses[i], jwst_exoplanet_names[i], fontsize = 14, rotation=45 )

# Plot location of Solar System planets (which will be replaced in post-processing by images):
plt.plot([0.387], [0.055], 'o', ms = 10, color = 'grey', zorder = 5) # Mercury
plt.plot([0.72], [0.815], 'o', ms = 10, color = 'orange', zorder = 5) # Venus
plt.plot([1.], [1.], 'o', ms = 10, color = 'blue', zorder = 5) # Earth
plt.plot([1.5], [0.107], 'o', ms = 10, color = 'red', zorder = 5) # Mars
plt.plot([9.58], [95.16], 'o', ms = 10, color = 'orange', zorder = 5) # Saturn
plt.plot([5.2], [317.906], 'o', ms = 10, color = 'green', zorder = 5) # Jupiter
plt.plot([19.2], [14.5], 'o', ms = 10, color = 'blue', zorder = 5) # Uranus
plt.plot([30.1], [17.1], 'o', ms = 10, color = 'cornflowerblue', zorder = 5) # Neptune


# Set labels, fontsizes:
plt.title('Exoplanets being targetted by JWST', fontsize=18, fontweight='bold')

plt.xlim(4e-3, 2e4)
plt.ylim(1e-2, 2e4)
plt.yticks([1e-1, 1, 10, 1e2, 1e3, 1e4], ['0.1', '1', '10', '100', '1,000', '10,000'], fontsize = 16)
plt.xticks([1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4], ['0.01','0.1', '1', '10', '100', '1,000', '10,000'], fontsize = 16)
plt.ylabel('Planet mass (M$_\oplus$)', fontsize = 18)
plt.xlabel('Orbital distance (AU)', fontsize = 18)
plt.tight_layout()
plt.savefig('espinoza_pre-figure2_'+cycle+'.pdf', dpi=300)
# Show the plot
plt.show()
