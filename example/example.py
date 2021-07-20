"""
small, ready-to-use sample script showing how to run SPART.

It shows to use cases:

    1) Run SPART with PROSPECT5-D as leaf model
    2) Run SPART with PROSPECT-PRO as leaf model
"""

import spart as SPART

# CASE 1: PROSPECT-D
# run leaf model (PROSPECT-D)
leafbio = SPART.LeafBiology(Cab=40, Cca=10, Cw=0.02, Cdm=0.01, Cs=0, Cant=10, N=1.5)

# run soil model (BSM)
soilpar = SPART.SoilParameters(B=0.5, lat=20, lon=50, SMp=15)

# run SAILH
canopy = SPART.CanopyStructure(LAI=3, LIDFa=-0.35, LIDFb=-0.15, q=0.05) # h = hot spot parameter

# define sun-observer geometry
angles = SPART.Angles(sol_angle=40, obs_angle=0, rel_angle=0)

# run atmosphere model (SMAC)
atm = SPART.AtmosphericProperties(aot550=0.3246, uo3=0.3480, uh2o=1.4116, Pa=1013.25)

# run model for specific sensor (in the example, Sentinel2A-MSI)
spart_s2_prospect5d = SPART.SPART(soilpar,
                                  leafbio,
                                  canopy,
                                  atm,
                                  angles,
                                  sensor='Sentinel2A-MSI',
                                  DOY=100)
results_s2_prospect5d = spart_s2_prospect5d.run()  # Pandas DataFrame containing R_TOC, R_TOA, L_TOA

# CASE 2: PROSPECT-PRO (affects change in leafbio, only
leafbio = SPART.LeafBiology(Cab=40, Cca=10, Cw=0.02, Cs=0, Cdm=0.0, Cant=10, N=1.5, PROT=0.0012, CBC=0.0035)

spart_s2_prospectpro = SPART.SPART(soilpar,
                                  leafbio,
                                  canopy,
                                  atm,
                                  angles,
                                  sensor='Sentinel2A-MSI',
                                  DOY=100)
results_s2_prospectpro = spart_s2_prospectpro.run()  # Pandas DataFrame containing R_TOC, R_TOA, L_TOA
