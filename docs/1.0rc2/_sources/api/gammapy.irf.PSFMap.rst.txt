PSFMap
======

.. currentmodule:: gammapy.irf

.. autoclass:: PSFMap
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~PSFMap.energy_name
      ~PSFMap.mask_safe_image
      ~PSFMap.psf_map
      ~PSFMap.required_axes
      ~PSFMap.tag

   .. rubric:: Methods Summary

   .. autosummary::

      ~PSFMap.containment
      ~PSFMap.containment_radius
      ~PSFMap.containment_radius_map
      ~PSFMap.copy
      ~PSFMap.cutout
      ~PSFMap.downsample
      ~PSFMap.from_gauss
      ~PSFMap.from_geom
      ~PSFMap.from_hdulist
      ~PSFMap.get_psf_kernel
      ~PSFMap.normalize
      ~PSFMap.peek
      ~PSFMap.plot_containment_radius_vs_energy
      ~PSFMap.plot_psf_vs_rad
      ~PSFMap.read
      ~PSFMap.sample_coord
      ~PSFMap.slice_by_idx
      ~PSFMap.stack
      ~PSFMap.to_hdulist
      ~PSFMap.to_image
      ~PSFMap.to_region_nd_map
      ~PSFMap.write

   .. rubric:: Attributes Documentation

   .. autoattribute:: energy_name
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
