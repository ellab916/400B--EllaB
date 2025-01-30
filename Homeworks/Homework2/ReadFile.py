"""
Created on Tue Jan 28 12:56:41 2025

@author: Ella Butler
"""
import numpy as np
import astropy.units as u

def Read(filename): 
    """
    Reads the MW_000.txt file and extracts the time, total number of 
    particles, and an array of data that stores the remainder of the file.
    
    
    Input: filename, which will be the Milky Way data file. 

    Outputs: 
        Time (float): in Myr
        Total number of particles (int)
        A data array storing the remainder of the file 
    -------
    """
    file = open(filename, 'r')     # opens file
    line1 = file.readline()        # reads the first line
    label, value = line1.split()   # 
    time = float(value) * u.Myr
    line2 = file.readline()        # reads the second lime 
    label2, value2 = line2.split()
    total = int(value2)
    file.close()                   # closes the file
    
    # read the remainder of the file
    data = np.genfromtxt(filename,dtype=None,names=True,skip_header=3)
    return time, total, data 



