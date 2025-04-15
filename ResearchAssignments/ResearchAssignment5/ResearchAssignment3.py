"""
Created on Mon Mar 31 20:15:20 2025

Author: Ella Butler
"""

# Q.  How do the stellar tidal tails that form during the first encounter of 
# MW/M31 evolve over the merger? 

# Methods: 
	# •	Identify a snapshot after the first encounter.  
	# •	Make a plot of the stellar disk distribution of the MW and M31 
    # (2D histogram) and visually identify the tidal tails you want to study. 
	# •	I would repeat steps  1 and 2 until you find a snapshot where the tidal
    # tail looks easily identifiable by eye.
	# •	Identify tidal tails – how will you do this?
	# • Make a phase diagram with stars from each galaxy 
    # (V vs R - see Lab 7) and identify outlier stars that deviate from the
    # rotation curve  and are at large radii.  
    # You can select the indices for those particles and see if those particles 
    # match the tidal tail in position space. 
	# •	Time evolution: 
	# •	If you are doing this analysis using the phase diagram, 
    # you can keep making phase diagrams and watch how the outlier stars 
    # evolve in time. You could compute the dispersion of those stars also. 

# Pseudocode for analyzing the evolution of stellar tidal tails 
# during the MW/M31 merger

# Load necessary libraries
import numpy as np
import astropy.units as u
from astropy.constants import G

# import plotting modules
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# import necessary modules
from ReadFile import Read
from CenterofMass import CenterOfMass # from Homework 4

# (from Lab 7)

# Create a COM of object for M31 Disk (particle type=2) 
COMD = CenterOfMass("M31_000.txt",2)

# Compute COM of M31 using disk particles
COMP = COMD.COM_P(0.1)
COMV = COMD.COM_V(COMP[0],COMP[1],COMP[2])

# Determine positions of disk particles relative to COM 
xD = COMD.x - COMP[0]
yD = COMD.y - COMP[1]
zD = COMD.z - COMP[2] 

# total magnitude
rtot = np.sqrt(xD**2 + yD**2 + zD**2)

# Determine velocities of disk particles relative to COM motion
vxD = COMD.vx - COMV[0].value
vyD = COMD.vy - COMV[1].value
vzD = COMD.vz - COMV[2].value

# total velocity 
vtot = np.sqrt(vxD**2 + vyD**2 + vzD**2)

# Arrays for r and v 
r = np.array([xD,yD,zD]).T # transposed 
v = np.array([vxD,vyD,vzD]).T

# M31 Disk Density 
fig, ax= plt.subplots(figsize=(12, 10))

# ADD HERE
# plot the particle density for M31 using a 2D histogram
# plt.hist2D(pos1,pos2, bins=, norm=LogNorm(), cmap='' )
# cmap options: 
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html  
#   e.g. 'magma', 'viridis'
# can modify bin number to make the plot smoother
plt.hist2d(xD, yD, bins=150, norm=LogNorm(), cmap='twilight')

cbar = plt.colorbar()
cbar.set_label("Number of disk particles per bin", fontsize=15)

# Add axis labels
plt.xlabel('x (kpc)', fontsize=22)
plt.ylabel('y (kpc)', fontsize=22)

#set axis limits
plt.ylim(-40,40)
plt.xlim(-40,40)

#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

def RotateFrame(posI,velI):
    """a function that will rotate the position and velocity vectors
    so that the disk angular momentum is aligned with z axis. 
    
    PARAMETERS
    ----------
        posI : `array of floats`
             3D array of positions (x,y,z)
        velI : `array of floats`
             3D array of velocities (vx,vy,vz)
             
    RETURNS
    -------
        pos: `array of floats`
            rotated 3D array of positions (x,y,z) 
            such that disk is in the XY plane
        vel: `array of floats`
            rotated 3D array of velocities (vx,vy,vz) 
            such that disk angular momentum vector
            is in the +z direction 
    """
    
    # compute the angular momentum
    L = np.sum(np.cross(posI,velI), axis=0)
    
    # normalize the angular momentum vector
    L_norm = L/np.sqrt(np.sum(L**2))


    # Set up rotation matrix to map L_norm to
    # z unit vector (disk in xy-plane)
    
    # z unit vector
    z_norm = np.array([0, 0, 1])
    
    # cross product between L and z
    vv = np.cross(L_norm, z_norm)
    s = np.sqrt(np.sum(vv**2))
    
    # dot product between L and z 
    c = np.dot(L_norm, z_norm)
    
    # rotation matrix
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v_x = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])
    R = I + v_x + np.dot(v_x, v_x)*(1 - c)/s**2

    # Rotate coordinate system
    pos = np.dot(R, posI.T).T
    vel = np.dot(R, velI.T).T
    
    return pos, vel

# compute the rotated position and velocity vectors
rn, vn = RotateFrame(r, v)

# Determine the positions of the disk particles in 
# cylindrical coordinates. (like in Lab 6)

cyl_r = np.sqrt(rn[:,0]**2 + rn[:,1]**2)              # radial
cyl_theta = np.arctan2(rn[:,1], rn[:,0]) * 180/np.pi  # theta in degrees

# Make a phase diagram of R vs Theta

fig = plt.figure(figsize=(12,10))
ax = plt.subplot(111)

# Plot 2D Histogram of r vs theta
# ADD HERE

plt.hist2d(cyl_r, cyl_theta, bins=150, norm=LogNorm(), cmap='twilight')
plt.colorbar()

# Add axis labels
plt.xlabel('R (kpc)', fontsize=22)
plt.ylabel(r'$\theta$ (deg)', fontsize=22)

#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

