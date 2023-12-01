SpatialModel
============

.. currentmodule:: gammapy.modeling.models

.. autoclass:: SpatialModel
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SpatialModel.covariance
      ~SpatialModel.default_parameters
      ~SpatialModel.evaluation_bin_size_min
      ~SpatialModel.evaluation_radius
      ~SpatialModel.evaluation_region
      ~SpatialModel.frozen
      ~SpatialModel.is_energy_dependent
      ~SpatialModel.parameters
      ~SpatialModel.phi_0
      ~SpatialModel.position
      ~SpatialModel.position_error
      ~SpatialModel.position_lonlat
      ~SpatialModel.type

   .. rubric:: Methods Summary

   .. autosummary::

      ~SpatialModel.__call__
      ~SpatialModel.copy
      ~SpatialModel.evaluate_geom
      ~SpatialModel.freeze
      ~SpatialModel.from_dict
      ~SpatialModel.from_parameters
      ~SpatialModel.from_position
      ~SpatialModel.integrate_geom
      ~SpatialModel.plot
      ~SpatialModel.plot_error
      ~SpatialModel.plot_grid
      ~SpatialModel.plot_interactive
      ~SpatialModel.plot_position_error
      ~SpatialModel.reassign
      ~SpatialModel.to_dict
      ~SpatialModel.unfreeze

   .. rubric:: Attributes Documentation

   .. autoattribute:: covariance
   .. autoattribute:: default_parameters
   .. autoattribute:: evaluation_bin_size_min
   .. autoattribute:: evaluation_radius
   .. autoattribute:: evaluation_region
   .. autoattribute:: frozen
   .. autoattribute:: is_energy_dependent
   .. autoattribute:: parameters
   .. autoattribute:: phi_0
   .. autoattribute:: position
   .. autoattribute:: position_error
   .. autoattribute:: position_lonlat
   .. autoattribute:: type

   .. rubric:: Methods Documentation

   .. automethod:: __call__
   .. automethod:: copy
   .. automethod:: evaluate_geom
   .. automethod:: freeze
   .. automethod:: from_dict
   .. automethod:: from_parameters
   .. automethod:: from_position
   .. automethod:: integrate_geom
   .. automethod:: plot
   .. automethod:: plot_error
   .. automethod:: plot_grid
   .. automethod:: plot_interactive
   .. automethod:: plot_position_error
   .. automethod:: reassign
   .. automethod:: to_dict
   .. automethod:: unfreeze
