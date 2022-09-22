from gammapy.maps import RegionGeom
geom = RegionGeom.create("icrs;circle(83.63, 22.01, 0.5)")
geom.plot_region()