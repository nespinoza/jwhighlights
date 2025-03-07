import matplotlib.pyplot as plt
import numpy as np
import os
import glob

import matplotlib.image as mpimg

def read_file(filename):

    out = {}
    f = open(filename, 'r')
    firstime = True
    while True:

        line = f.readline()
        
        if line != '':

            if line[0] != '#':

                if not firstime:

                    data = line.split(',')
                    for i in range(len(keys)):

                        try:

                            out[keys[i]] = np.append( out[keys[i]], float(data[i]) )

                        except:

                            out[keys[i]] = np.append( out[keys[i]], data[i] )

                elif firstime:

                    keys = line.split(',')
                    for i in range(len(keys)):

                        if i != len(keys) - 1:

                            out[keys[i]] = np.array([])

                        else:

                            keys[i] = keys[i][:-2]
                            out[keys[i]] = np.array([])

                    firstime = False
            

        else:

            break


    f.close()
    return out

def add_image_to_plot(filename, x, y, size):

    img = mpimg.imread(filename)
