RecoPSFMap
==========

.. currentmodule:: gammapy.irf

.. autoclass:: RecoPSFMap
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~RecoPSFMap.energy_name
      ~RecoPSFMap.has_single_spatial_bin
      ~RecoPSFMap.mask_safe_image
      ~RecoPSFMap.psf_map
      ~RecoPSFMap.required_axes
      ~RecoPSFMap.tag

   .. rubric:: Methods Summary

   .. autosummary::

      ~RecoPSFMap.containment
      ~RecoPSFMap.containment_radius
      ~RecoPSFMap.containment_radius_map
      ~RecoPSFMap.copy
      ~RecoPSFMap.cutout
      ~RecoPSFMap.downsample
      ~RecoPSFMap.from_gauss
      ~RecoPSFMap.from_geom
      ~RecoPSFMap.from_hdulist
      ~RecoPSFMap.get_psf_kernel
      ~RecoPSFMap.normalize
      ~RecoPSFMap.peek
      ~RecoPSFMap.plot_containment_radius_vs_energy
      ~RecoPSFMap.plot_psf_vs_rad
      ~RecoPSFMap.read
      ~RecoPSFMap.sample_coord
      ~RecoPSFMap.slice_by_idx
      ~RecoPSFMap.stack
      ~RecoPSFMap.to_hdulist
      ~RecoPSFMap.to_image
      ~RecoPSFMap.to_region_nd_map
      ~RecoPSFMap.write

   .. rubric:: Attributes Documentation

   .. autoattribute:: energy_name
   .. autoattribute:: has_single_spatial_bin
   .. autoattribute:: mask_safe_image
   .. autoattribute:: psf_map
   .. autoattribute:: required_axes
   .. autoattribute:: tag

   .. rubric:: Methods Documentation

   .. automethod:: containment
   .. automethod:: containment_radius
   .. automethod:: containment_radius_map
   .. automethod:: copy
   .. automethod:: cutout
   .. automethod:: downsample
   .. automethod:: from_gauss
   .. automethod:: from_geom
   .. automethod:: from_hdulist
   .. automethod:: get_psf_kernel
   .. automethod:: normalize
   .. automethod:: peek
   .. automethod:: plot_containment_radius_vs_energy
   .. automethod:: plot_psf_vs_rad
   .. automethod:: read
   .. automethod:: sample_coord
   .. automethod:: slice_by_idx
   .. automethod:: stack
   .. automethod:: to_hdulist
   .. automethod:: to_image
   .. automethod:: to_region_nd_map
   .. automethod:: write
