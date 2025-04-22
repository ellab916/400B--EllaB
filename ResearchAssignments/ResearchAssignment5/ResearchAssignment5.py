"""
Created on Mon Mar 31 17:32:21 2025

@author: ellabutler
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

# ---------------------------

# import the built-in os module for interacting with the operating system
import os

# define full paths to the folders that contain the .txt files 
# and loop through to combine folder path with filename to get 
# the full absolute path
m31_folder_path = '/Users/ellabutler/400B--EllaB/M31'
for filename in os.listdir(m31_folder_path):
    m31_full_path = os.path.join(m31_folder_path, filename)
    
mw_folder_path = '/Users/ellabutler/400B--EllaB/MW'
for filename in os.listdir(mw_folder_path):
    mw_full_path = os.path.join(mw_folder_path, filename)

# load necessary libraries
import numpy as np
import astropy.units as u
from astropy.constants import G

# import plotting modules
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# import necessary modules
from ReadFile import Read
from CenterofMass import CenterOfMass # from homework 4

# (from lab 7)

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


    # set up rotation matrix to map L_norm to
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

    # rotate coordinate system
    pos = np.dot(R, posI.T).T
    vel = np.dot(R, velI.T).T
    
    return pos, vel

# create a COM of object for M31 Disk (particle type=2) 
COMD = CenterOfMass("/Users/ellabutler/400B--EllaB/MW/MW_600.txt", 2)

# compute COM of M31 using disk particles
COMP = COMD.COM_P(0.1)
COMV = COMD.COM_V(COMP[0],COMP[1],COMP[2])

# determine positions of disk particles relative to COM 
xD = COMD.x - COMP[0]
yD = COMD.y - COMP[1]
zD = COMD.z - COMP[2] 

# total magnitude
rtot = np.sqrt(xD**2 + yD**2 + zD**2)

# determine velocities of disk particles relative to COM motion
vxD = COMD.vx - COMV[0].value
vyD = COMD.vy - COMV[1].value
vzD = COMD.vz - COMV[2].value

# total velocity 
vtot = np.sqrt(vxD**2 + vyD**2 + vzD**2)

# arrays for r and v 
r = np.array([xD,yD,zD]).T # transposed 
v = np.array([vxD,vyD,vzD]).T

# m31 disk density 
fig, ax= plt.subplots(figsize=(12, 10))

# plot the particle density for m31 using a 2d histogram
plt.hist2d(xD, yD, bins=150, norm=LogNorm(), cmap='twilight')
cbar = plt.colorbar()
cbar.set_label("Number of disk particles per bin", fontsize=15)

# add axis labels
plt.xlabel('x (kpc)', fontsize=22)
plt.ylabel('y (kpc)', fontsize=22)

# set axis limits
plt.ylim(-90,90)
plt.xlim(-90,90)

# adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# compute the rotated position and velocity vectors
rn, vn = RotateFrame(r, v)

# determine the positions of the disk particles in 
# cylindrical coordinates (like in Lab 6)

cyl_r = np.sqrt(rn[:,0]**2 + rn[:,1]**2)              # radial
cyl_theta = np.arctan2(rn[:,1], rn[:,0]) * 180/np.pi  # theta in degrees

# make a phase diagram of r vs theta

fig = plt.figure(figsize=(12,10))
ax = plt.subplot(111)

# plot 2d histogram of r vs theta

plt.hist2d(cyl_r, cyl_theta, bins=150, norm=LogNorm(), cmap='twilight')
plt.colorbar()

# add axis labels
plt.xlabel('R (kpc)', fontsize=22)
plt.ylabel(r'$\theta$ (deg)', fontsize=22)

# adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# Set snapshot range
snapshots = np.arange(200, 450, 5)

# Arrays for M31
time_m31 = []
dispersion_m31 = []

# Arrays for MW
time_mw = []
dispersion_mw = []

# --- Loop over snapshots ---
for snap in snapshots:
    # --- M31 ---
    filename_m31 = f'/Users/ellabutler/400B--EllaB/M31/M31_{snap:03d}.txt'
    COMD_m31 = CenterOfMass(filename_m31, 2)

    COMP_m31 = COMD_m31.COM_P(0.1)
    COMV_m31 = COMD_m31.COM_V(COMP_m31[0], COMP_m31[1], COMP_m31[2])

    xD = COMD_m31.x - COMP_m31[0]
    yD = COMD_m31.y - COMP_m31[1]
    zD = COMD_m31.z - COMP_m31[2]
    vxD = COMD_m31.vx - COMV_m31[0].value
    vyD = COMD_m31.vy - COMV_m31[1].value
    vzD = COMD_m31.vz - COMV_m31[2].value

    r = np.array([xD, yD, zD]).T
    v = np.array([vxD, vyD, vzD]).T

    rn, vn = RotateFrame(r, v)

    cyl_r = np.sqrt(rn[:,0]**2 + rn[:,1]**2)
    cyl_theta = np.arctan2(rn[:,1], rn[:,0]) * 180/np.pi

    tail_indices = (cyl_r > 50) & (   #  best radius limit? 
        ((cyl_theta > 50) & (cyl_theta < 150)) |
        ((cyl_theta > -150) & (cyl_theta < -50))
    )

    v_tail = v[tail_indices]

    if len(v_tail) > 0:
        sigma_v = np.std(np.linalg.norm(v_tail, axis=1))
    else:
        sigma_v = np.nan

    time_m31.append(COMD_m31.time)
    dispersion_m31.append(sigma_v)

    # --- Milky Way (MW) ---
    filename_mw = f'/Users/ellabutler/400B--EllaB/MW/MW_{snap:03d}.txt'
    COMD_mw = CenterOfMass(filename_mw, 2)

    COMP_mw = COMD_mw.COM_P(0.1)
    COMV_mw = COMD_mw.COM_V(COMP_mw[0], COMP_mw[1], COMP_mw[2])

    xD = COMD_mw.x - COMP_mw[0]
    yD = COMD_mw.y - COMP_mw[1]
    zD = COMD_mw.z - COMP_mw[2]
    vxD = COMD_mw.vx - COMV_mw[0].value
    vyD = COMD_mw.vy - COMV_mw[1].value
    vzD = COMD_mw.vz - COMV_mw[2].value

    r = np.array([xD, yD, zD]).T
    v = np.array([vxD, vyD, vzD]).T

    rn, vn = RotateFrame(r, v)

    cyl_r = np.sqrt(rn[:,0]**2 + rn[:,1]**2)
    cyl_theta = np.arctan2(rn[:,1], rn[:,0]) * 180/np.pi

    tail_indices = (cyl_r > 30) & (
        ((cyl_theta > 50) & (cyl_theta < 150)) |
        ((cyl_theta > -150) & (cyl_theta < -50))
    )

    v_tail = v[tail_indices]

    if len(v_tail) > 0:
        sigma_v = np.std(np.linalg.norm(v_tail, axis=1))
    else:
        sigma_v = np.nan

    time_mw.append(COMD_mw.time)
    dispersion_mw.append(sigma_v)
    
# Convert to arrays
time_m31 = np.array(time_m31)
dispersion_m31 = np.array(dispersion_m31)

time_mw = np.array(time_mw)
dispersion_mw = np.array(dispersion_mw)

# Plot both
plt.figure(figsize=(10,7))

plt.plot(time_m31/1000, dispersion_m31, label='M31', marker='o', color='blue')
plt.plot(time_mw/1000, dispersion_mw, label='Milky Way', marker='s', color='red')

plt.xlabel('Time (Gyr)', fontsize=18) # is units of Gyr preferred? 
plt.ylabel('Velocity Dispersion (km/s)', fontsize=18)
plt.title('Tidal Tail Velocity Dispersion Comparison', fontsize=20)
plt.grid()
plt.legend(fontsize=16)
plt.show()

# add error bars? especially when there's lower amounts of stars? 
