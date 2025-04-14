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
	# •	You could compute the Jacobi Radius for each galaxy as a proxy for the 
    # tidal radius and choose particles outside that. 
	# •	Or you could make a phase diagram with stars from each galaxy 
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

# import plotting modules
import matplotlib.pyplot as plt

# import necessary modules
from ReadFile import Read
from CenterofMass import CenterOfMass

# (from Lab 7)

# Load in high resolution simulation data
# Create a COM of object for M31 Disk (particle type=2) 
# (Using Code from Homework 4)
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


# Load in simulation data
def load_simulation_data(snapshot_id):
    # Placeholder for loading particle data (positions, velocities, masses, etc.)
    return particle_data 

# Function to plot stellar disk distribution
def plot_stellar_disk(particle_data):
    # Create a 2D histogram of star positions to visualize tidal tails
    plt.hist2d(particle_data['x'], particle_data['y'], bins=100, cmap='inferno')
    plt.colorbar(label='Star Count')
    plt.xlabel('X [kpc]')
    plt.ylabel('Y [kpc]')
    plt.title('Stellar Disk Distribution')
    plt.show()

# Identify tidal tails using phase diagram method
def identify_tidal_tails(particle_data):
    # Construct phase space diagram (V vs R)
    velocities = particle_data['v']
    radii = particle_data['r']
    plot_phase_diagram(velocities, radii)
    # Identify outliers based on deviation from rotation curve
    tidal_tail_indices = identify_outlier_stars(velocities, radii)
    return tidal_tail_indices

# Function to plot phase diagram
def plot_phase_diagram(velocities, radii):
    plt.scatter(radii, velocities, s=1, alpha=0.5)
    plt.xlabel('Radius [kpc]')
    plt.ylabel('Theta')
    plt.title('Phase Diagram (Theta vs R)')
    plt.show()

# Function to identify outlier stars from rotation curve
def identify_outlier_stars(velocities, radii):
    # Placeholder logic for identifying outliers
    # For example, stars far from the median velocity at a given radius
    # You can refine this with statistical or model-based thresholds
    median_velocity = np.median(velocities)
    deviation = np.abs(velocities - median_velocity)
    threshold = np.percentile(deviation, 95)
    return np.where(deviation > threshold)

# Function to track the time evolution of tidal tails
def track_tidal_tail_evolution(initial_snapshot, final_snapshot, tidal_tail_indices):
    for snapshot in range(initial_snapshot, final_snapshot + 1):
        particle_data = load_simulation_data(snapshot)
        # Extract positions of previously identified tidal tail stars
        tail_positions = particle_data['position'][tidal_tail_indices]
        # Plot evolution
        plt.scatter(tail_positions[:,0], tail_positions[:,1], s=1, alpha=0.5)
        plt.xlabel('X [kpc]')
        plt.ylabel('Y [kpc]')
        plt.title(f'Tidal Tail Evolution - Snapshot {snapshot}')
        plt.show()

# Main execution
initial_snapshot = 100  # Example snapshot after first encounter
final_snapshot = 200    # Example final snapshot for evolution study

particle_data = load_simulation_data(initial_snapshot)
plot_stellar_disk(particle_data)

tidal_tail_indices = identify_tidal_tails(particle_data)
track_tidal_tail_evolution(initial_snapshot, final_snapshot, tidal_tail_indices)
