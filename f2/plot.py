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

# Retrieve data for all exoplanets:
all_exoplanets = read_file('documents/PSCompPars_2025.02.05_16.15.06.csv')

# Retrieve properties for planets with measured masses and orbital distances:
idx = np.where( (all_exoplanets['pl_bmasse'] != '') & (all_exoplanets['pl_orbsmax'] != '') )[0]
all_exoplanet_masses = all_exoplanets['pl_bmasse'][idx].astype('float')
all_exoplanet_distances = all_exoplanets['pl_orbsmax'][idx].astype('float')

# Now retrieve properties for exoplanets observed by JWST:
jwst_exoplanets = read_file('documents/all.csv')

# Compile name of unique planets:
jwst_exoplanet_names = list(set(jwst_exoplanets['Planet']))
jwst_direct_imaging = np.ones(len(jwst_exoplanet_names))
jwst_exoplanet_masses = np.zeros(len(jwst_exoplanet_names))
jwst_exoplanet_distances = np.zeros(len(jwst_exoplanet_names))

# Get properties for these:
for i in range(len(jwst_exoplanet_names)):

    jwst_exoplanet = jwst_exoplanet_names[i]
    
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
plt.plot(all_exoplanet_distances, all_exoplanet_masses, '.', color = 'silver')

# Plot JWST exoplanets. Identify direct imaging exoplanets:
idx_direct_imaging = np.where(jwst_direct_imaging == 1.)[0]
idx_non_direct_imaging = np.where(jwst_direct_imaging == 0.)[0]

# Plot all planets:
#plt.plot(jwst_exoplanet_distances, jwst_exoplanet_masses, 'h', mfc = '#ce943a', mec = 'black', ms = 10)

# Plot directly imaged ones:
plt.plot(jwst_exoplanet_distances[idx_direct_imaging], 
         jwst_exoplanet_masses[idx_direct_imaging], 
         'h', mfc = 'gold', mec = 'black', ms = 10)

# Plot non-directly imaged ones:
plt.plot(jwst_exoplanet_distances[idx_non_direct_imaging], 
         jwst_exoplanet_masses[idx_non_direct_imaging], 
         'h', mfc = 'darkorange', mec = 'black', ms = 10) 

# Plot a special color for PSR J2322-2650, which is neither transiting not "directly imaged"
idx = np.where( np.array(jwst_exoplanet_names) == 'PSR J2322-2650 b' )[0]
plt.plot([jwst_exoplanet_distances[idx]],
         [jwst_exoplanet_masses[idx]],
         'h', mfc = 'peachpuff', mec = 'black', ms = 10)
         

# Plot location of Solar System planets (which will be replaced in post-processing by images):
plt.plot([0.387], [0.055], 'o', ms = 10, color = 'grey') # Mercury
plt.plot([0.72], [0.815], 'o', ms = 10, color = 'orange') # Venus
plt.plot([1.], [1.], 'o', ms = 10, color = 'blue') # Earth
plt.plot([1.5], [0.107], 'o', ms = 10, color = 'red') # Mars
plt.plot([9.58], [95.16], 'o', ms = 10, color = 'orange') # Saturn
plt.plot([5.2], [317.906], 'o', ms = 10, color = 'green') # Jupiter
plt.plot([19.2], [14.5], 'o', ms = 10, color = 'blue') # Uranus
plt.plot([30.1], [17.1], 'o', ms = 10, color = 'cornflowerblue') # Neptune


# Set labels, fontsizes:
plt.title('Exoplanets being targetted by JWST', fontsize=18, fontweight='bold')

plt.yticks([1e-1, 1, 10, 1e2, 1e3, 1e4], ['0.1', '1', '10', '100', '1,000', '10,000'], fontsize = 16)
plt.xticks([1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4], ['0.01','0.1', '1', '10', '100', '1,000', '10,000'], fontsize = 16)
plt.ylabel('Planet mass (M$_\oplus$)', fontsize = 18)
plt.xlabel('Orbital distance (AU)', fontsize = 18)
plt.tight_layout()
plt.savefig('espinoza_pre-figure2.pdf', dpi=300)
# Show the plot
plt.show()
