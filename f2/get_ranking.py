import numpy as np
import os
import glob

from utils import read_file

def rank_stars(thedict):

    # Go object by object, and compile total time per unique star name:
    unique_stars = list(set(thedict['Star']))

    # Count time per star:
    total_time = np.zeros(len(unique_stars))
    for i in range(len(unique_stars)):

        star = unique_stars[i]

        for j in range(len(thedict['Star'])):

            if thedict['Star'][j] == star and thedict['Target multiplier'][j] != 0:
        
                total_time[i] += thedict['Seconds on target'][j]

    idx = np.argsort(total_time)[::-1]
    unique_stars = np.array(unique_stars)[idx]
    total_time = total_time[idx]

    return unique_stars, total_time

files = glob.glob('documents/*')

print('\n Which file?\n')
for i in range(len(files)):

    print('\t ({0:}) {1:}'.format(i, files[i].split('/')[-1]))

print('\n')
choice = input('Enter selection > ')
choice2 = int(input('Rank the most popular n stars, where n is: '))
file = files[int(choice)]

thedict = read_file(file)
ranked_stars, ranked_time = rank_stars(thedict)

for i in range(choice2):

    print('\t {0:}. {1:} ({2:.2f} hours).'.format(str(i), ranked_stars[i], ranked_time[i]/3600.),)
