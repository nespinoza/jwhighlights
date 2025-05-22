import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('ticks')

import mplcursors
import matplotlib as mpl
from matplotlib.image import imread
import matplotlib.transforms as transforms

from astropy.constants import G
import astropy.units as q

from utils import read_file

def get_escape_velocity(M, R):
    """
    Given mass M and radius R of a planet (or set of planets), return the escape velocity in km/s.

    M:  Mass of the planet (in masses of the Earth).
    R:  Radius of the planet (in Earth radii).

    """

    return np.sqrt( 2. * G * ( M * q.Mearth ) / ( R * q.Rearth ) ).to(q.km / q.s).value

def get_IXUV(Rstar, Tstar, a):
    """
    Given a stellar radius R, effective temperature T and distance to the planet a, calculates the cummulative XUV flux with respect 
    to the Solar one using the scaling relationship in Zahnle & Catling (2017; https://arxiv.org/abs/1702.03386).

    Rstar:  Stellar radius (solar radii).
    Tstar:  Stellar effective temperature (in Kelvins).
    a:      Distance to the planet (in AU).
    """

    # Now compute XUV cummulative irradiation. Before contiuning, define Teff for the Sun as per IAU's resolution I don't remember:
    Teff_sun = 5772 * q.K 

    # First, estimate luminosities of the stars (modulo Boltzmann's constant and factor of 4*pi as we'll normalize to the Sun's):
    Lstar = ( ( Rstar * q.Rsun )**2 ) * ( ( Tstar * q.K )**4 )
    Lsun = ( ( 1. * q.Rsun )**2 ) * ( ( Teff_sun * q.K )**4 )

    # Now I/Isun:
    I = ( Lstar.value / Lsun.value ) * ( ( 1. )**2 / ( a )**2 )

    # And IXUV:
    return I * ( Lstar.value / Lsun.value )**(-0.6)

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# Retrieve data for all exoplanets:
all_exoplanets = read_file('../f2/documents/PSCompPars_2025.02.05_16.15.06.csv')

# Retrieve properties for planets with measured masses and orbital distances:
idx = np.where( (all_exoplanets['pl_bmasse'] != '') & (all_exoplanets['pl_orbsmax'] != '') & (all_exoplanets['pl_rade'] != '') & \
                (all_exoplanets['st_rad'] != '') & (all_exoplanets['st_teff'] != '') )[0]

# Names
all_exoplanet_names = all_exoplanets['pl_name'][idx]
# Needed for escape velocity
all_exoplanet_radii = all_exoplanets['pl_rade'][idx].astype('float')
all_exoplanet_masses = all_exoplanets['pl_bmasse'][idx].astype('float')
# Needed to compute integrated XUV:
all_stellar_radii = all_exoplanets['st_rad'][idx].astype('float')
all_stellar_teff = all_exoplanets['st_teff'][idx].astype('float')
all_exoplanet_distances = all_exoplanets['pl_orbsmax'][idx].astype('float')

# Now retrieve properties for exoplanets observed by JWST:
jwst_exoplanets = read_file('../f2/documents/all.csv')

# Compile name of unique planets:
jwst_exoplanet_names = np.array(list(set(jwst_exoplanets['Planet'])))
jwst_exoplanet_radii = np.zeros(len(jwst_exoplanet_names))
jwst_exoplanet_masses = np.zeros(len(jwst_exoplanet_names))
jwst_exoplanet_stellar_radii = np.zeros(len(jwst_exoplanet_names))
jwst_exoplanet_stellar_teff = np.zeros(len(jwst_exoplanet_names))
jwst_exoplanet_distances = np.zeros(len(jwst_exoplanet_names))

# Get properties for these:
for i in range(len(jwst_exoplanet_names)):

    jwst_exoplanet = jwst_exoplanet_names[i]
    
    for j in range(len(jwst_exoplanets['Planet'])):

        if jwst_exoplanet == jwst_exoplanets['Planet'][j]:

            try:

                jwst_exoplanet_masses[i] = jwst_exoplanets['Planet Mass (Earth masses)'][j]
                jwst_exoplanet_radii[i] = jwst_exoplanets['Planet Radius (Earth radii)'][j]
                jwst_exoplanet_stellar_radii[i] = jwst_exoplanets['Stellar Radius (Solar Radii)'][j]
                jwst_exoplanet_stellar_teff[i] = jwst_exoplanets['Stellar Teff (K)'][j]
                jwst_exoplanet_distances[i] = jwst_exoplanets['Planet semi-major axis (AU)'][j]

            except:

                #print(jwst_exoplanet, 'failed one of the properties; PID ', 
                #      jwst_exoplanets['PID'][j], 
                #      'Cycle', jwst_exoplanets['Cycle'][j])
                jwst_exoplanet_masses[i] = 99.#jwst_exoplanets['Planet Mass (Earth masses)'][j]
                jwst_exoplanet_radii[i] = 99.#jwst_exoplanets['Planet Radius (Earth radii)'][j]
                jwst_exoplanet_stellar_radii[i] = 99.#jwst_exoplanets['Stellar Radius (Solar Radii)'][j]
                jwst_exoplanet_stellar_teff[i] = 99.#jwst_exoplanets['Stellar Teff (K)'][j]
                jwst_exoplanet_distances[i] = 99.#jwst_exoplanets['Planet semi-major axis (AU)'][j]

            break

idx = np.where(jwst_exoplanet_masses != 99.)[0]
for x in [jwst_exoplanet_names, jwst_exoplanet_radii, jwst_exoplanet_masses, jwst_exoplanet_stellar_radii, jwst_exoplanet_stellar_teff, jwst_exoplanet_distances]:

    x = x[idx]

# Calculate escape velocity for all planets in km/s:
escape_velocity = get_escape_velocity(all_exoplanet_masses, all_exoplanet_radii)

# Same for jwst exoplanets:
jwst_escape_velocity = get_escape_velocity(jwst_exoplanet_masses, jwst_exoplanet_radii) 

# Now compute XUV cummulative irradiation:
Ixuv = get_IXUV(all_stellar_radii, all_stellar_teff, all_exoplanet_distances)#I * ( Lstar.value / Lsun.value )**(-0.6)
jwst_Ixuv = get_IXUV(jwst_exoplanet_stellar_radii, jwst_exoplanet_stellar_teff, jwst_exoplanet_distances)

# Check that cross-matched properties match between JWEL and ours. 
# If not, use ours, not because mines are "better" but because I checked them manually at a given point in time (March 2025), and thus
# it's a "frozen version"):
for i in range( len(jwst_exoplanet_names) ):

    idx = np.where(jwst_exoplanet_names[i] == all_exoplanet_names)[0]
    if (Ixuv[idx] != jwst_Ixuv[i]) or (escape_velocity[idx] != jwst_escape_velocity[i]):

        print(jwst_exoplanet_names[i])
        print(jwst_Ixuv[i], jwst_escape_velocity[i])
        print(Ixuv[idx][0], escape_velocity[idx][0])
        print('Properties for JWEL estimated on ',jwst_exoplanet_names[i], \
              ' ({0:.2f}, {1:.2f})'.format(jwst_Ixuv[i], jwst_escape_velocity[i]), \
              'didnt match exoplanet archive ones', ' ({0:.2f}, {1:.2f})'.format(Ixuv[idx][0], escape_velocity[idx][0]))

        Ixuv[idx] = jwst_Ixuv[i]
        escape_velocity[idx] = jwst_escape_velocity[i]


# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# Define scale at the very beggining:
plt.xscale('log')
plt.yscale('log')

# Plot only planets smaller than 2 Earth-radii:
idx_all = np.where( (all_exoplanet_radii < 2.) & (all_stellar_teff < 4000.) )
idx_jwst = np.where( (jwst_exoplanet_radii < 2.) & (jwst_exoplanet_stellar_teff < 4000.) )

jwst_exoplanet_names = jwst_exoplanet_names[idx_jwst]
all_exoplanet_names = all_exoplanet_names[idx_all]

x = np.linspace(0.1,200,20000)
I_x = 6.191017244909004 # instellation at bottom of CS in Zanle & atling
ve_x = 9.382264934578679 # escape velocity at crossing
constant = I_x / (ve_x**4)
# constant = (1e-6/(0.18**4))

y = constant*(x**4)

ax.plot(x,y,'-', color = 'cornflowerblue', lw = 10, alpha = 0.5)

# Plot all the exoplanets:
escape_velocity, Ixuv, all_exoplanet_masses, all_exoplanet_radii = escape_velocity[idx_all], Ixuv[idx_all], all_exoplanet_masses[idx_all], all_exoplanet_radii[idx_all]
scatter1 = ax.plot(escape_velocity, Ixuv, '.', color = 'silver')

for i in range(len(all_exoplanet_names)):

    if '406.01' in all_exoplanet_names[i]:

        print(all_exoplanet_names[i], escape_velocity[i], Ixuv[i], all_exoplanet_masses[i], all_exoplanet_radii[i])
        ax.plot(escape_velocity[i], Ixuv[i], 'v', color = 'green', ms = 10)

# Plot JWST exoplanets:
jwst_escape_velocity, jwst_Ixuv, jwst_exoplanet_masses, jwst_exoplanet_radii = jwst_escape_velocity[idx_jwst], jwst_Ixuv[idx_jwst], jwst_exoplanet_masses[idx_jwst], jwst_exoplanet_radii[idx_jwst]
scatter2 = ax.plot(jwst_escape_velocity,
            jwst_Ixuv,
            'h', mfc = 'darkorange', mec = 'black', ms = 10) 

# Plot location of Solar System planets (which will be replaced in post-processing by images):
plt.plot([4.25], [6.59], 'o', ms = 10, color = 'grey') # Mercury
plt.plot([10.36], [1.86], 'o', ms = 10, color = 'orange') # Venus
plt.plot([11.2], [1.], 'o', ms = 10, color = 'blue') # Earth
plt.plot([5.03], [0.421], 'o', ms = 10, color = 'red') # Mars

# Set labels, fontsizes:
plt.title('Exoplanets being targetted by JWST', fontsize=18, fontweight='bold')
plt.xlim(4,25)
plt.ylim(0.1,10500)
plt.xticks([4, 5, 6, 7, 8, 9, 10, 20], ['4', '5', '6', '7', '8', '9', '10', '20'], fontsize = 16)
plt.yticks([0.1, 1, 10, 1e2, 1e3, 1e4], ['0.1', '1', '10', '100', '1,000', '10,000'], fontsize = 16)
#plt.yticks([1e-1, 1, 10, 1e2, 1e3, 1e4], ['0.1', '1', '10', '100', '1,000', '10,000'], fontsize = 16)
#plt.xticks([1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4], ['0.01','0.1', '1', '10', '100', '1,000', '10,000'], fontsize = 16)
plt.ylabel('I$_{XUV}$ (Cummulative XUV radiation)', fontsize = 18)
plt.xlabel('Escape velocity (km/s)', fontsize = 18)
plt.tight_layout()

def update_annot1(sel):

    xx,yy = sel.target
    idx = np.where((escape_velocity==xx) & (Ixuv==yy))[0]
    mass, radius = all_exoplanet_masses[idx][0], all_exoplanet_radii[idx][0]
    sel.annotation.set_text(all_exoplanet_names[idx][0] + ' | {0:.2f}ME, {1:.2f}RE'.format(mass, radius))

def update_annot2(sel):

    xx,yy = sel.target
    idx = np.where((jwst_escape_velocity==xx) & (jwst_Ixuv==yy))[0]
    mass, radius = jwst_exoplanet_masses[idx][0], jwst_exoplanet_radii[idx][0]
    sel.annotation.set_text(jwst_exoplanet_names[idx][0] + ' | {0:.2f}ME, {1:.2f}RE'.format(mass, radius))

cursor1 = mplcursors.cursor(scatter1, hover=True)
cursor1.connect("add", update_annot1)

cursor2 = mplcursors.cursor(scatter2, hover=True)
cursor2.connect("add", update_annot2)

# Show the plot
plt.show()
