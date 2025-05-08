"""
Created on Sat Apr 21 22:22:08 2025

AUTHOR: Ella Butler
"""

# My research project is to look at stellar tidal tails that form during the 
# first encounter of the MW/M31 merger and plot the velocity dispersion of the
# stars within the tidal tails that formed. 

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

# ---------------------------

# import the built-in os module for interacting with the operating system
import os

# load necessary libraries
import numpy as np

# import plotting modules
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# import class modules
from ReadFile import Read
from CenterofMass import CenterOfMass


def FullPaths(folder_path):
    """
    Generates a list of full file paths for all files in a specified folder.

    PARAMETERS
    ----------
    folder_path : str
        Path to the folder containing the files.

    RETURNS
    -------
    full_paths : list of str
        List containing the full paths to each file in the folder.
    """
    # list all entries in the folder and joins each with the folder path
    # to create the full path to the file
    return [os.path.join(folder_path, filename) for filename in 
                                        os.listdir(folder_path)]

# function from Lab 7
def RotateFrame(posI, velI):
    """
    Rotates position and velocity vectors so the disk angular 
    momentum vector aligns with the z-axis.

    PARAMETERS
    ----------
    posI : ndarray
        2D array of particle positions, shape (N, 3).
    velI : ndarray
        2D array of particle velocities, shape (N, 3).

    RETURNS
    -------
    pos : ndarray
        Rotated 2D array of particle positions after alignment (shape (N, 3)).
    vel : ndarray
        Rotated 2D array of particle velocities after alignment (shape (N, 3)).
    """
    # compute total angular momentum vector of the system
    L = np.sum(np.cross(posI, velI), axis=0)
    
    # normalize the angular momentum vector
    L_norm = L / np.sqrt(np.sum(L**2))
    
    # define the target axis (z-axis) for alignment
    z_norm = np.array([0, 0, 1])
    
    # compute the rotation axis using the cross product between L and z
    vv = np.cross(L_norm, z_norm)
    
    # compute sine and cosine of the angle between L and z
    s = np.sqrt(np.sum(vv**2))    # sin(theta)
    c = np.dot(L_norm, z_norm)    # cos(theta)
    
    # construct the skew-symmetric cross-product matrix of vv
    v_x = np.array([
        [0, -vv[2], vv[1]],
        [vv[2], 0, -vv[0]],
        [-vv[1], vv[0], 0]
    ])
    
    # construct the rotation matrix- rodrigues' rotation formula
    I = np.identity(3)
    R = I + v_x + np.dot(v_x, v_x) * (1 - c) / s**2
    
    # rotate positions and velocities using the rotation matrix
    pos = np.dot(R, posI.T).T
    vel = np.dot(R, velI.T).T
    return pos, vel



# from HW 4
def LoadCOMData(filename, ptype=2):
    """
    Loads the center of mass (COM) position, velocity, and disk data 
    from a .txt file.

    PARAMETERS
    ----------
    filename : str
        Path to the .txt file containing COM and disk data.

    RETURNS
    -------
    COMD : object
        A structured array containing center of mass disk data, 
        including time attribute.
    COMP : ndarray
        2D array of particle positions relative to the COM (shape (N, 3)).
    COMV : ndarray
        2D array of particle velocities relative to the COM (shape (N, 3)).
    """
    # CenterOfMass object for the given file and particle type
    COMD = CenterOfMass(filename, ptype)
    
    # get center of mass position using a specified tolerance (e.g., 0.1 kpc)
    COMP = COMD.COM_P(0.1)
    
    # compute velocity of center of mass using the COM position
    COMV = COMD.COM_V(COMP[0], COMP[1], COMP[2])
    
    return COMD, COMP, COMV



# from HW 4
def DiskData(COMD, COMP, COMV):
    """
    Prepares the disk data by shifting particle positions and velocities 
    relative to the center of mass (COM).

    PARAMETERS
    ----------
    COMD : object
        Center of mass data, with attributes such as position and velocity.
    COMP : ndarray
        2D array of original particle positions (shape (N, 3)).
    COMV : ndarray
        2D array of original particle velocities (shape (N, 3)).

    RETURNS
    -------
    r : ndarray
        2D array of particle positions shifted to the center of mass frame 
        (shape (N, 3)).
    v : ndarray
        2D array of particle velocities shifted to the center of mass frame 
        (shape (N, 3)).
    """
    
    # shift particle positions by subtracting the COM position components
    xD = COMD.x - COMP[0]
    yD = COMD.y - COMP[1]
    zD = COMD.z - COMP[2]
    
    # shift particle velocities by subtracting COM velocity components
    vxD = COMD.vx - COMV[0].value
    vyD = COMD.vy - COMV[1].value
    vzD = COMD.vz - COMV[2].value
    
    # transpose coordinate system 
    r = np.array([xD, yD, zD]).T 
    v = np.array([vxD, vyD, vzD]).T
    
    return r, v



# from Lab 7
def DiskDensity(x, y, title="Disk Particle Density"):
    """
    Plots a 2D histogram of disk particle density in the xy-plane.

    PARAMETERS
    ----------
    x : ndarray
        1D array of x-coordinates of disk particles (units: kpc).
    y : ndarray
        1D array of y-coordinates of disk particles (units: kpc).
    title : str, optional
        Title for the plot (default is "Disk Particle Density").

    RETURNS
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the density plot.
    
    """
    
    # create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # plot a 2D histogram of particle positions with logarithmic color scaling
    plt.hist2d(x, y, bins=150, norm=LogNorm(), cmap='twilight')
    
    # add a colorbar to show particle counts per bin
    cbar = plt.colorbar()
    cbar.set_label("Number of disk particles per bin", fontsize=15)
    
    # label axes
    plt.xlabel('x (kpc)', fontsize=22)
    plt.ylabel('y (kpc)', fontsize=22)
    
    # set limits for plot axes
    plt.xlim(-90, 90)
    plt.ylim(-90, 90)
    
    # set font size for tick labels 
    label_size = 22
    matplotlib.rcParams['xtick.labelsize'] = label_size 
    matplotlib.rcParams['ytick.labelsize'] = label_size
    
    # set plot title, save to a .png file, and show plot
    plt.title(title, fontsize=20)
    plt.savefig('DiskDensity')
    plt.show()


    
# from Lab 7
def PhaseDiagram(rn, galaxy_name):
    """
    Plots a phase diagram of disk particles in cylindrical coordinates (R, θ)
    for a given galaxy snapshot.

    PARAMETERS
    ----------
    rn : ndarray
        2D array of particle positions (shape: N x 3), 
        assumed to be in Cartesian coordinates.
    galaxy_name : str
        Name of the galaxy used for the plot title.

    RETURNS
    -------
    None.
        Displays and saves a 2D histogram plot of cylindrical radius vs. angle.

    """
    
    # convert Cartesian x, y positions to cylindrical radius
    cyl_r = np.sqrt(rn[:,0]**2 + rn[:,1]**2)
    
    # Compute polar angle theta in degrees (arctan of y/x)
    cyl_theta = np.arctan2(rn[:,1], rn[:,0]) * 180/np.pi
    
    # create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # plot 2D histogram of R vs. theta with logarithmic color scale
    plt.hist2d(cyl_r, cyl_theta, bins=150, norm=LogNorm(), cmap='twilight')
    
    # add a colorbar to show particle density per bin
    plt.colorbar()
    
    # label axes
    plt.xlabel('R (kpc)', fontsize=22)
    plt.ylabel(r'$\theta$ (deg)', fontsize=22)
    
    # set tick label sizes
    label_size = 22
    matplotlib.rcParams['xtick.labelsize'] = label_size 
    matplotlib.rcParams['ytick.labelsize'] = label_size
    
    # set plot title, save to a .png file, and show plot
    plt.title(f"Phase Diagram of {galaxy_name} Disk at Snap 350", fontsize=20)
    plt.savefig('PhaseDiagram')
    plt.show()



# original function 
def VelocityDispersion(folder_path, tail_radius_cut, snapshots, label):
    """
    Computes the velocity dispersion of tidal tail particles 
    over a series of simulation snapshots.

    PARAMETERS
    ----------
    folder_path : str
        Path to the folder containing the simulation data files.
    tail_radius_cut : float
        Minimum cylindrical radius (kpc) to define particles 
        considered part of the tidal tail.
    snapshots : array-like
        Sequence of snapshot numbers to process 
        (e.g., np.arange(0, 800, 10)).
    label : str
        Prefix label for the filenames (e.g., 'M31' or 'MW').

    RETURNS
    -------
    times : ndarray
        Array of times corresponding to each snapshot 
        (units assumed to be Myr unless otherwise specified).
    dispersions : ndarray
        Array of velocity dispersion values (km/s) 
        for tidal tail particles at each snapshot.
    """
    
    # preallocate arrays to store time and velocity dispersion for each snapshot
    times = np.zeros(len(snapshots))
    dispersions = np.zeros(len(snapshots))
    
    # loop over each snapshot
    for i, snap in enumerate(snapshots):
        
        # construct the filename for the current snapshot
        filename = f'{folder_path}/{label}_{snap:03d}.txt'
        
        # load COM and particle data for the snapshot
        COMD, COMP, COMV = LoadCOMData(filename)
        
        # shift positions and velocities to COM frame
        r, v = DiskData(COMD, COMP, COMV)
        
        # rotate coordinate frame so angular momentum points along z-axis
        rn, vn = RotateFrame(r, v)
        
        # convert to cylindrical coordinates R and theta
        cyl_r = np.sqrt(rn[:, 0]**2 + rn[:, 1]**2)
        cyl_theta = np.degrees(np.arctan2(rn[:, 1], rn[:, 0]))
        
        # select particles in the tidal tail based on radius and angle cuts
        tail_indices = (cyl_r > tail_radius_cut) & (
            ((cyl_theta > 50) & (cyl_theta < 150)) |
            ((cyl_theta > -150) & (cyl_theta < -50))
        )
        
        # extract velocities of particles in the tidal tail
        v_tail = v[tail_indices]
        
        # compute velocity dispersion if tail particles exist
        if v_tail.size > 0:
            sigma_v = np.std(np.linalg.norm(v_tail, axis=1))
        else:
            sigma_v = np.nan  # if no particles found, assign NaN

        # store time and dispersion into preallocated arrays
        times[i] = COMD.time
        dispersions[i] = sigma_v

    return times, dispersions



# original function
def DispersionEvolution(time_m31, disp_m31, time_mw, disp_mw):
    """
    Plots the evolution of velocity dispersion over time for M31 and 
    the Milky Way tidal tails.

    PARAMETERS
    ----------
    time_m31 : ndarray
        1D array of times corresponding to M31 tidal tail snapshots 
        (units: Myr).
    disp_m31 : ndarray
        1D array of velocity dispersions for M31 at each time step 
        (units: km/s).
    time_mw : ndarray
        1D array of times corresponding to Milky Way tidal tail snapshots 
        (units: Myr).
    disp_mw : ndarray
        1D array of velocity dispersions for the Milky Way at each time step 
        (units: km/s).

    RETURNS
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the dispersion evolution plot.
    """
    # create a new figure with specified size
    plt.figure(figsize=(17,10))
    
    # plot M31 dispersion data, converting time from Myr to Gyr
    plt.plot(time_m31/1000, disp_m31, label='M31', marker='o', color='blue')
    
    # plot MW dispersion data, converting time from Myr to Gyr
    plt.plot(time_mw/1000, disp_mw, label='Milky Way', marker='s', color='red')
    
    # set axis labels and title 
    plt.xlabel('Time (Gyr)', fontsize=18)
    plt.ylabel('Velocity Dispersion (km/s)', fontsize=18)
    plt.title('Velocity Dispersion of Tidal Tails', fontsize=20)
    
    # add grid
    plt.grid()
    
    # add legend to distinguish between M31 and MW data
    plt.legend(fontsize=16)
    
    # save to a file and display plot
    plt.savefig('VelocityDispersion')
    plt.show()
    


# original function
def process_galaxy(galaxy_name, folder_path, snap_num):
    """
    Processes a single galaxy snapshot by generating a disk density plot 
    and a phase diagram.

    PARAMETERS
    ----------
    galaxy_name : str
        Name of the galaxy (e.g., 'MW' or 'M31').
    folder_path : str
        Path to the folder containing the snapshot data files.
    snap_num : int
        Snapshot number to process.

    RETURNS
    -------
    None.
    """
    
    # load COM, position, and velocity data
    comD, comP, comV = LoadCOMData(f"{folder_path}/{galaxy_name}_{snap_num}.txt")
    
    # get disk particle data
    r, v = DiskData(comD, comP, comV)
    
    # make disk density plot
    xD, yD, zD = r[:,0], r[:,1], r[:,2]
    DiskDensity(xD, yD, title=f"{galaxy_name} Disk Density at Snap {snap_num}")
    
    # rotate and plot phase diagram
    rn, vn = RotateFrame(r, v)
    
    # plot and save phase diagram
    PhaseDiagram(rn, galaxy_name)

# define folder paths to the simulation data for M31 and MW 
m31_folder_path = '/Users/ellabutler/400B--EllaB/M31'
mw_folder_path = '/Users/ellabutler/400B--EllaB/MW'

# generate and save disk density and phase diagram plots 
# for MW and M31 at snapshot 350
process_galaxy("MW", mw_folder_path, 350)
process_galaxy("M31", m31_folder_path, 350)

# define range of snapshot numbers to analyze 
# full graph- every 10 steps from 0 to 800- fig size 20, 15
# first encounter- 250-360- fig size 17, 10
snapshots = np.arange(250, 360, 10)

# compute velocity dispersion over time for the M31 tidal tail
# using particles beyond a cylindrical radius of 50 kpc
time_m31, disp_m31 = VelocityDispersion(m31_folder_path, 
                            50, snapshots, label="M31")

# compute velocity dispersion over time for the MW tidal tail
# using particles beyond a cylindrical radius of 30 kpc
time_mw, disp_mw = VelocityDispersion(mw_folder_path, 
                            30, snapshots, label="MW")

# plot and save the velocity dispersion evolution over time for both galaxies
DispersionEvolution(time_m31, disp_m31, time_mw, disp_mw)
    

