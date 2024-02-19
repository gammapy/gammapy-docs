MapAxes
=======

.. currentmodule:: gammapy.maps

.. autoclass:: MapAxes
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~MapAxes.center_coord
      ~MapAxes.is_flat
      ~MapAxes.is_unidimensional
      ~MapAxes.iter_with_reshape
      ~MapAxes.names
      ~MapAxes.primary_axis
      ~MapAxes.reverse
      ~MapAxes.shape

   .. rubric:: Methods Summary

   .. autosummary::

      ~MapAxes.assert_names
      ~MapAxes.bin_volume
      ~MapAxes.coord_to_idx
      ~MapAxes.coord_to_pix
      ~MapAxes.copy
      ~MapAxes.count
      ~MapAxes.downsample
      ~MapAxes.drop
      ~MapAxes.from_default
      ~MapAxes.from_table
      ~MapAxes.from_table_hdu
      ~MapAxes.get_coord
      ~MapAxes.index
      ~MapAxes.index_data
      ~MapAxes.is_allclose
      ~MapAxes.pad
      ~MapAxes.pix_to_coord
      ~MapAxes.pix_to_idx
      ~MapAxes.rename_axes
      ~MapAxes.replace
      ~MapAxes.resample
      ~MapAxes.slice_by_idx
      ~MapAxes.squash
      ~MapAxes.to_header
      ~MapAxes.to_table
      ~MapAxes.to_table_hdu
      ~MapAxes.upsample

   .. rubric:: Attributes Documentation

   .. autoattribute:: center_coord
   .. autoattribute:: is_flat
   .. autoattribute:: is_unidimensional
   .. autoattribute:: iter_with_reshape
   .. autoattribute:: names
   .. autoattribute:: primary_axis
   .. autoattribute:: reverse
   .. autoattribute:: shape

   .. rubric:: Methods Documentation

   .. automethod:: assert_names
   .. automethod:: bin_volume
   .. automethod:: coord_to_idx
   .. automethod:: coord_to_pix
   .. automethod:: copy
   .. automethod:: count
   .. automethod:: downsample
   .. automethod:: drop
   .. automethod:: from_default
   .. automethod:: from_table
   .. automethod:: from_table_hdu
   .. automethod:: get_coord
   .. automethod:: index
   .. automethod:: index_data
   .. automethod:: is_allclose
   .. automethod:: pad
   .. automethod:: pix_to_coord
   .. automethod:: pix_to_idx
   .. automethod:: rename_axes
   .. automethod:: replace
   .. automethod:: resample
   .. automethod:: slice_by_idx
   .. automethod:: squash
   .. automethod:: to_header
   .. automethod:: to_table
   .. automethod:: to_table_hdu
   .. automethod:: upsample
