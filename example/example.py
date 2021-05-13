import spart as SPART

# run leaf model (PROSPECT-D)
leafbio = SPART.LeafBiology(Cab=40, Cca=10, Cw=0.02, Cdm=0.01, Cs=0, Cant=10, N=1.5)

# run soil model (BSM)
soilpar = SPART.SoilParameters(B=0.5, lat=0, lon=100, SMp=15)

# run SAILH
canopy = SPART.CanopyStructure(LAI=3, LIDFa=-0.35, LIDFb=-0.15, q=0.05)

# define sun-observer geometry
angles = SPART.Angles(sol_angle=40, obs_angle=0, rel_angle=0)

# run atmosphere model (SMAC)
atm = SPART.AtmosphericProperties(aot550=0.3246, uo3=0.3480, uh2o=1.4116, Pa=1013.25)

# run model for specific sensor (in the example, MODIS)
spart_s2 = SPART.SPART(soilpar, leafbio, canopy, atm, angles, sensor='Sentinel2A-MSI', DOY=100)
results_s2 = spart_s2.run()  # Pandas DataFrame containing R_TOC, R_TOA, L_TOA
