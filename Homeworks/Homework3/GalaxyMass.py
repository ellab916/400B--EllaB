"""
Created on Tue Feb 4 16:41:21 2025

author: Ella Butler
"""
import numpy as np
from ReadFile import Read

def ComponentMass(filename, ptype): 
    """
    Calculates the total mass of a specified galaxy component.
    
    Inputs:
    filename (str)- The name of the file containing particle data.
    ptype (int)- The particle type: 1 (Halo), 2 (Disk), 3 (Bulge).
    
    Outputs:
    total_mass (float)- The total mass in units of 10^12 M☉, 
    rounded to three decimal places.
    """
    time, total_particles, data = Read(filename)
    
    
    # Filter particles of the given type
    particle_mask = data['type'] == ptype
    
    mass = data['m']
    mass_masked = mass[particle_mask]
    
    # Sums the mass of the selected particles (measured in 1e10 MSun)
    # and converts result into 1e12 MSun
    total_mass = (np.sum(mass_masked) * 1e10) / 1e12
    
    
    return np.round(total_mass, 3)


gals = np.array(["MW_000.txt", "M31_000.txt", "M33_000.txt"])

for gal in gals:

    halo = ComponentMass(gal, 1)
    disk = ComponentMass(gal, 2)
    bulge = ComponentMass(gal, 3)
    
    print(f"{gal[:3]} Halo Mass: {halo} [1e12 MSun]")
    print(f"{gal[:3]} Disk Mass: {disk} [1e12 MSun]")
    print(f"{gal[:3]} Bulge Mass: {bulge} [1e12 MSun]")
    print(f"{gal[:3]} Total Mass: {halo+disk+bulge} [1e12 MSun]")
    print("---------------")


