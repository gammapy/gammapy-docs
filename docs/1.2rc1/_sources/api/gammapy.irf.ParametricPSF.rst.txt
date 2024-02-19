ParametricPSF
=============

.. currentmodule:: gammapy.irf

.. autoclass:: ParametricPSF
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~ParametricPSF.axes
      ~ParametricPSF.data
      ~ParametricPSF.default_interp_kwargs
      ~ParametricPSF.fov_alignment
      ~ParametricPSF.has_offset_axis
      ~ParametricPSF.is_pointlike
      ~ParametricPSF.quantity
      ~ParametricPSF.required_axes
      ~ParametricPSF.required_parameters
      ~ParametricPSF.tag
      ~ParametricPSF.unit

   .. rubric:: Methods Summary

   .. autosummary::

      ~ParametricPSF.containment
      ~ParametricPSF.containment_radius
      ~ParametricPSF.cumsum
      ~ParametricPSF.evaluate
      ~ParametricPSF.evaluate_containment
      ~ParametricPSF.evaluate_direct
      ~ParametricPSF.evaluate_parameters
      ~ParametricPSF.from_hdulist
      ~ParametricPSF.from_table
      ~ParametricPSF.info
      ~ParametricPSF.integral
      ~ParametricPSF.integrate_log_log
      ~ParametricPSF.interp_missing_data
      ~ParametricPSF.is_allclose
      ~ParametricPSF.normalize
      ~ParametricPSF.pad
      ~ParametricPSF.peek
      ~ParametricPSF.plot_containment_radius
      ~ParametricPSF.plot_containment_radius_vs_energy
      ~ParametricPSF.plot_psf_vs_rad
      ~ParametricPSF.read
      ~ParametricPSF.slice_by_idx
      ~ParametricPSF.to_hdulist
      ~ParametricPSF.to_psf3d
      ~ParametricPSF.to_table
      ~ParametricPSF.to_table_hdu
      ~ParametricPSF.to_unit
      ~ParametricPSF.write

   .. rubric:: Attributes Documentation

   .. autoattribute:: axes
   .. autoattribute:: data
   .. autoattribute:: default_interp_kwargs
   .. autoattribute:: fov_alignment
   .. autoattribute:: has_offset_axis
   .. autoattribute:: is_pointlike
   .. autoattribute:: quantity
   .. autoattribute:: required_axes
   .. autoattribute:: required_parameters
   .. autoattribute:: tag
   .. autoattribute:: unit

   .. rubric:: Methods Documentation

   .. automethod:: containment
   .. automethod:: containment_radius
   .. automethod:: cumsum
   .. automethod:: evaluate
   .. automethod:: evaluate_containment
   .. automethod:: evaluate_direct
   .. automethod:: evaluate_parameters
   .. automethod:: from_hdulist
   .. automethod:: from_table
   .. automethod:: info
   .. automethod:: integral
   .. automethod:: integrate_log_log
   .. automethod:: interp_missing_data
   .. automethod:: is_allclose
   .. automethod:: normalize
   .. automethod:: pad
   .. automethod:: peek
   .. automethod:: plot_containment_radius
   .. automethod:: plot_containment_radius_vs_energy
   .. automethod:: plot_psf_vs_rad
   .. automethod:: read
   .. automethod:: slice_by_idx
   .. automethod:: to_hdulist
   .. automethod:: to_psf3d
   .. automethod:: to_table
   .. automethod:: to_table_hdu
   .. automethod:: to_unit
   .. automethod:: write
