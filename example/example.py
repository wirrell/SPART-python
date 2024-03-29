"""
small, ready-to-use sample script showing how to run SPART.
"""
import SPART

# Use default values from original SPART paper

# Define leaf biology
leafbio = SPART.LeafBiology(Cab=40, Cca=10, Cw=0.02, Cdm=0.01, Cs=0, Cant=10, N=1.5)

# Define soil parameters
soilpar = SPART.SoilParameters(B=0.5, lat=0, lon=100, SMp=20)

# Define canopy structure
canopy = SPART.CanopyStructure(LAI=3, LIDFa=-0.35, LIDFb=-0.15, q=0.05)

# Define sun-observer geometry
angles = SPART.Angles(sol_angle=40, obs_angle=0, rel_angle=0)

# Define atmospheric properties
atm = SPART.AtmosphericProperties(aot550=0.325, uo3=0.35, uh2o=1.41, Pa=1013.25)

# run model for specific sensor and DOY (in the example, Sentinel2A-MSI)
spart_s2_prospect5d = SPART.SPART(soilpar,
                                  leafbio,
                                  canopy,
                                  atm,
                                  angles,
                                  sensor='Sentinel2A-MSI',
                                  DOY=100)
results_s2_prospect5d = spart_s2_prospect5d.run()  # Pandas DataFrame containing R_TOC, R_TOA, L_TOA
