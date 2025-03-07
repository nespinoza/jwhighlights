import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('ticks')

import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# Data taken from our study of all the Cycle's data:
x = np.array([0, 1, 2])  # X-axis (e.g., time or categories)
y1 = np.array([242.1, 357.57, 227.75]) # Disk science
y2 = np.array([11.1+152.9, 57.33+267.97, 95.48+265.41]) # Direct image science
y3 = np.array([648.3, 658.62, 753.5])  # Transit science
y4 = np.array([110.7+17.9, 13.38+115.11, 127.5+46.46]) # Eclipse science
y5 = np.array([220.6, 222.07, 74.68]) # Phase curves

# Stack the elements
y = np.vstack([y1, y2, y3, y4, y5])

# Plotting
plt.figure(figsize=(10, 6))
# Colors for each layer
colors = ['#E07A5F', '#F2CC8F', '#C4C6E7', '#454372', '#2F2963']
# Labels for each element
plt.stackplot(x, y, colors=colors, alpha=0.8)
# Plot total line:
yall = np.sum(y, axis = 0)

# Plot thick line on top:
plt.plot(x, yall, color = 'black', linewidth = 7)

# Set ranges:
plt.xlim(0,2)

# Set labels, fontsizes:
plt.title('JWST Allocated hours to Exoplanet Science', fontsize=18, fontweight='bold')
plt.xticks([0, 1, 2], ['Cycle 1', 'Cycle 2', 'Cycle 3'], fontsize = 16)
plt.yticks(fontsize = 16)
plt.ylabel('Hours', fontsize=18)
plt.tight_layout()
plt.savefig('espinoza_pre-figure1.pdf', dpi=300)
# Show the plot
plt.show()
