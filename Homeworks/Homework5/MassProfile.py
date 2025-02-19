"""
Created on Mon Feb 17 12:51:00 2025

Author: Ella Butler
"""
import numpy as np
from ReadFile import Read
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.constants import G
from CenterofMass import CenterOfMass


# Set G to the required units
G = G.to(u.kpc * u.km**2 / u.s**2 / u.Msun)


class MassProfile:
    def __init__(self, galaxy, snap):
        """
        Initialize the MassProfile class.
        
        Parameters:
        galaxy (str): Galaxy name ("MW", "M31", or "M33")
        snap (int): Snapshot number (e.g., 0, 1, ...)
        """
        
        # Store the galaxy name as a class attribute
        self.gname = galaxy
        
        # Construct the filename
        ilbl = '000' + str(snap)
        ilbl = ilbl[-3:]  # Keep only the last 3 digits
        self.filename = f"{galaxy}_{ilbl}.txt"
        
        # Read the data using Read from ReadFile
        self.time, self.total, self.data = Read(self.filename)
        
        # Extract positions and masses from the data
        self.x = self.data['x'] * u.kpc
        self.y = self.data['y'] * u.kpc
        self.z = self.data['z'] * u.kpc
        self.m = self.data['m']  # We'll assign units to mass later
        
        print(f"Loaded data for {self.gname} from {self.filename}")
        
        
    def MassEnclosed(self, ptype, radii):
        """
        Calculate the mass enclosed within a given radius.
    
        Parameters:
            ptype (int): Particle type (1 = Halo, 2 = Disk, 3 = Bulge)
            radii (array): Array of radii (1D array) in kpc
    
        Returns:
            array: Mass enclosed within each radius (in units of Msun)
        """
        # Create a CenterOfMass object to find the COM position
        COM = CenterOfMass(self.filename, 2)  # Use Disk particles for COM
        COM_pos = COM.COM_P(0.1) * u.kpc  # Convert COM position to a 
        # Quantity with kpc units
        COMpos = COM.COM_P(0.1)  # Get COM position
    
    # Extract positions and masses for the specified particle type
        index = np.where(self.data['type'] == ptype)
        xP = self.x[index] - COM_pos[0]  # Centered x position
        yP = self.y[index] - COM_pos[1]  # Centered y position
        zP = self.z[index] - COM_pos[2]  # Centered z position
        mP = self.m[index] * 1e10 * u.Msun  # Convert mass to Msun
        
    
    # Initialize the enclosed mass array
        enclosed_mass = np.zeros(len(radii)) * u.Msun
    
    # Loop over the radius array
        for i in range(len(radii)):
            # Calculate the distance of each particle from COM
            r = np.sqrt(xP**2 + yP**2 + zP**2)
        
            # Find particles within the current radius
            index_within = np.where(r < radii[i] * u.kpc)
        
            # Sum the masses of those particles
            enclosed_mass[i] = np.sum(mP[index_within])
    
        return enclosed_mass
    
    
    def MassEnclosedTotal(self, radii):
        """
        Calculate the total mass enclosed within a given radius 
        for all components.
    
        Parameters:
        radii (array): Array of radii (1D array) in kpc
    
        Returns:
        array: Total mass enclosed within each radius (in units of Msun)
        """
        # Calculate enclosed mass for each component
        halo_mass = self.MassEnclosed(1, radii)  # Halo particles
        disk_mass = self.MassEnclosed(2, radii)  # Disk particles
    
        # Check if the galaxy is M33 (which has no bulge)
        if self.gname == "M33":
            bulge_mass = np.zeros(len(radii)) * u.Msun  # No bulge
        else:
            bulge_mass = self.MassEnclosed(3, radii)  # Bulge particles
    
        # Sum up all components to get total enclosed mass
        total_mass = halo_mass + disk_mass + bulge_mass
    
        return total_mass
    
    
    def HernquistMass(self, r, a, Mhalo):
        """
        Calculate the mass enclosed within a given radius using 
        the Hernquist profile.
    
        Parameters:
            r (array or float): Radius or array of radii (in kpc)
            a (float): Scale factor (in kpc)
            Mhalo (float): Total halo mass (in Msun)
    
        Returns:
            array: Hernquist enclosed mass at each radius (in units of Msun)
        """
        # Calculate Hernquist enclosed mass
        mHern = Mhalo * (r**2) / ((a + r)**2)
    
        # Return the mass in units of Msun
        return mHern * u.Msun
    
    
    def CircularVelocity(self, ptype, radii):
        """
        Calculate the circular velocity for a given component.
    
        Parameters:
            ptype (int): Particle type (1 = Halo, 2 = Disk, 3 = Bulge)
            radii (array): Array of radii (1D array) in kpc
    
        Returns:
            array: Circular velocities at each radius (in units of km/s)
        """
        # Get the enclosed mass for the given particle type
        M_enclosed = self.MassEnclosed(ptype, radii)
    
        # Calculate the circular velocity using the formula
        Vcirc = np.sqrt(G * M_enclosed / (radii * u.kpc))
    
        # Return the velocities rounded to two decimal places
        return np.around(Vcirc, 2)
    
    
    def CircularVelocityTotal(self, radii):
        """
        Calculate the total circular velocity for all components.
    
        Parameters:
            radii (array): Array of radii (1D array) in kpc
    
        Returns:
            array: Total circular velocities at each radius (in units of km/s)
        """
        # Calculate circular velocities for each component
        Vhalo = self.CircularVelocity(1, radii)  # Halo particles
        Vdisk = self.CircularVelocity(2, radii)  # Disk particles
    
        # Check if the galaxy is M33 (which has no bulge)
        if self.gname == "M33":
            Vbulge = np.zeros(len(radii)) * u.km / u.s  # No bulge
        else:
            Vbulge = self.CircularVelocity(3, radii)  # Bulge particles
    
        # Calculate the total circular velocity
        Vtotal = np.sqrt(Vhalo**2 + Vdisk**2 + Vbulge**2)
    
        # Return the total velocities rounded to two decimal places
        return np.around(Vtotal, 2)
    
    
    def HernquistVCirc(self, r, a, Mhalo):
        """
        Calculate the circular speed using the Hernquist mass profile.
    
        Parameters:
            r (array or float): Radius or array of radii (in kpc)
            a (float): Scale factor (in kpc)
            Mhalo (float): Total halo mass (in Msun)
    
        Returns:
            array: Hernquist circular speed at each radius (in units of km/s)
        """
        # Get the enclosed mass using the Hernquist profile
        M_enclosed = self.HernquistMass(r, a, Mhalo)
    
        # Calculate the circular velocity using the formula
        Vcirc = np.sqrt(G * M_enclosed / (r * u.kpc))
    
        # Return the velocities rounded to two decimal places
        return np.around(Vcirc, 2)
    
def PlotMassProfile(galaxy_obj, a, Mhalo):
    """
    Plot the mass profile for each component and the total mass profile.
    
    Parameters:
        galaxy_obj (MassProfile): An instance of the MassProfile class
        a (float): Scale factor for the Hernquist profile (in kpc)
        Mhalo (float): Total halo mass for the Hernquist profile (in Msun)
    """
    # Define an array of radii from 0.25 to 30.5 kpc in steps of 0.5 kpc
    r = np.arange(0.25, 30.5, 0.5)
    
    # Calculate the mass profiles for each component
    halo_mass = galaxy_obj.MassEnclosed(1, r)
    disk_mass = galaxy_obj.MassEnclosed(2, r)
    
    # Check if the galaxy is M33 (which has no bulge)
    if galaxy_obj.gname == "M33":
        bulge_mass = np.zeros(len(r)) * u.Msun  # No bulge
    else:
        bulge_mass = galaxy_obj.MassEnclosed(3, r)
    
    # Calculate the total mass profile
    total_mass = galaxy_obj.MassEnclosedTotal(r)
    
    # Calculate the Hernquist profile for the dark matter halo
    hernquist_mass = galaxy_obj.HernquistMass(r, a, Mhalo)
    
    # Plot the mass profiles
    plt.figure(figsize=(10, 7))
    plt.semilogy(r, halo_mass, color='blue', linestyle='-', label='Halo')
    plt.semilogy(r, disk_mass, color='red', linestyle='--', label='Disk')
    if galaxy_obj.gname != "M33":
        plt.semilogy(r, bulge_mass, color='green', linestyle='-.', label='Bulge')
    plt.semilogy(r, total_mass, color='black', linestyle='-', linewidth=2, label='Total')
    plt.semilogy(r, hernquist_mass, color='purple', linestyle=':', label='Hernquist Fit')
    
    # Label the best fit scale length
    plt.text(20, 1e11, f'a = {a} kpc', color='purple')
    
    # Labels and legend
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Log(Mass Enclosed) (Msun)')
    plt.title(f'Mass Profile of {galaxy_obj.gname}')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

def PlotRotationCurve(galaxy_obj, a, Mhalo):
    """
    Plot the rotation curve for each component and the total circular velocity.
    
    Parameters:
    galaxy_obj (MassProfile): An instance of the MassProfile class
    a (float): Scale factor for the Hernquist profile (in kpc)
    Mhalo (float): Total halo mass for the Hernquist profile (in Msun)
    """
    # Define an array of radii from 0.25 to 30.5 kpc in steps of 0.5 kpc
    r = np.arange(0.25, 30.5, 0.5)
    
    # Calculate the circular velocities for each component
    halo_vcirc = galaxy_obj.CircularVelocity(1, r)
    disk_vcirc = galaxy_obj.CircularVelocity(2, r)
    
    # Check if the galaxy is M33 (which has no bulge)
    if galaxy_obj.gname == "M33":
        bulge_vcirc = np.zeros(len(r)) * u.km / u.s  # No bulge
    else:
        bulge_vcirc = galaxy_obj.CircularVelocity(3, r)
    
    # Calculate the total circular velocity
    total_vcirc = galaxy_obj.CircularVelocityTotal(r)
    
    # Calculate the Hernquist circular speed
    hernquist_vcirc = galaxy_obj.HernquistVCirc(r, a, Mhalo)
    
    # Plot the rotation curves
    plt.figure(figsize=(10, 7))
    plt.plot(r, halo_vcirc, color='blue', linestyle='-', label='Halo')
    plt.plot(r, disk_vcirc, color='red', linestyle='--', label='Disk')
    if galaxy_obj.gname != "M33":
        plt.plot(r, bulge_vcirc, color='green', linestyle='-.', label='Bulge')
    plt.plot(r, total_vcirc, color='black', linestyle='-', linewidth=2, label='Total')
    plt.plot(r, hernquist_vcirc, color='purple', linestyle=':', label='Hernquist Fit')
    
    # Label the best fit scale length
    plt.text(20, 100, f'a = {a} kpc', color='purple')
    
    # Labels and legend
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Circular Velocity (km/s)')
    plt.title(f'Rotation Curve of {galaxy_obj.gname}')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

# Define the galaxies and their respective halo masses
galaxies = np.array(["MW", "M31", "M33"])  # Array of galaxy names
Mhalos = np.array([1.975e12, 1.921e12, 0.187e12])  # Galaxy halo masses (Msun)
snap = 0  # Example snapshot number
a = 60  # Scale length (kpc)

# Loop through each galaxy and its corresponding halo mass
for i in range(len(galaxies)):
    galaxy_obj = MassProfile(galaxies[i], snap)  # Creates MassProfile instance
    
    # Generate plots for mass profile and rotation curve
    PlotMassProfile(galaxy_obj, a, Mhalos[i])
    PlotRotationCurve(galaxy_obj, a, Mhalos[i])

    

    




    
    
    
    
    





