EDispMap
========

.. currentmodule:: gammapy.irf

.. autoclass:: EDispMap
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~EDispMap.edisp_map
      ~EDispMap.mask_safe_image
      ~EDispMap.required_axes
      ~EDispMap.tag

   .. rubric:: Methods Summary

   .. autosummary::

      ~EDispMap.copy
      ~EDispMap.cutout
      ~EDispMap.downsample
      ~EDispMap.from_diagonal_response
      ~EDispMap.from_geom
      ~EDispMap.from_hdulist
      ~EDispMap.get_edisp_kernel
      ~EDispMap.normalize
      ~EDispMap.peek
      ~EDispMap.read
      ~EDispMap.sample_coord
      ~EDispMap.slice_by_idx
      ~EDispMap.stack
      ~EDispMap.to_edisp_kernel_map
      ~EDispMap.to_hdulist
      ~EDispMap.to_region_nd_map
      ~EDispMap.write

   .. rubric:: Attributes Documentation

   .. autoattribute:: edisp_map
   .. autoattribute:: mask_safe_image
   .. autoattribute:: required_axes
   .. autoattribute:: tag

   .. rubric:: Methods Documentation

   .. automethod:: copy
   .. automethod:: cutout
   .. automethod:: downsample
   .. automethod:: from_diagonal_response
   .. automethod:: from_geom
   .. automethod:: from_hdulist
   .. automethod:: get_edisp_kernel
   .. automethod:: normalize
   .. automethod:: peek
   .. automethod:: read
   .. automethod:: sample_coord
   .. automethod:: slice_by_idx
   .. automethod:: stack
   .. automethod:: to_edisp_kernel_map
   .. automethod:: to_hdulist
   .. automethod:: to_region_nd_map
   .. automethod:: write
