"""
Created on Wed Jan 29 12:26:02 2025

@author: Ella Butler
"""

import numpy as np
import astropy.units as u
from ReadFile import Read

def ParticleInfo(filename, ptype, pnum):
    """
    Computes distance, velocity magnitude, and mass for a given particle.
    
    Inputs:
    filename (str): The name of the file to read.
    ptype (int): Particle type (1=Dark Matter, 2=Disk Stars, 3=Bulge Stars)
    pnum (int): Index of the particle
    
    Outputs:
    Distance (float): 3D distance in kpc (rounded to 3 decimal places)
    velocity (float): 3D velocity in km/s (rounded to 3 decimal places)
    mass (float): Mass in solar masses
    """
    time, total_particles, data = Read(MW_000.txt) # reads the file
    
    index = np.where(data['type'] == ptype) # filters for given particle type
    
    x, y, z = data['x'][index][pnum], data['y'][index][pnum], data['z'][index][pnum]
    vx, vy, vz = data['vx'][index][pnum], data['vy'][index][pnum], data['vz'][index][pnum]
    mass = data['m'][index][pnum] * 1e10  # converts to solar masses
    
    # computes 3D distance and velocity magnitude
    distance = np.sqrt(x**2 + y**2 + z**2) * u.kpc
    velocity = np.sqrt(vx**2 + vy**2 + vz**2) * (u.km / u.s)
    
    # rounds values
    distance = np.around(distance.value, 3)
    velocity = np.around(velocity.value, 3)
    
    return distance, velocity, mass