from gammapy.maps import RegionNDMap
region_map = RegionNDMap.create("icrs;circle(83.63, 22.01, 0.5)")
region_map.plot_region()