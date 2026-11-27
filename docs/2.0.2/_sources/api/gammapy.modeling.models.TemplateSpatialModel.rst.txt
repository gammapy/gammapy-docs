TemplateSpatialModel
====================

.. currentmodule:: gammapy.modeling.models

.. autoclass:: TemplateSpatialModel
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~TemplateSpatialModel.covariance
      ~TemplateSpatialModel.default_parameters
      ~TemplateSpatialModel.evaluation_bin_size_min
      ~TemplateSpatialModel.evaluation_radius
      ~TemplateSpatialModel.evaluation_region
      ~TemplateSpatialModel.frozen
      ~TemplateSpatialModel.is_energy_dependent
      ~TemplateSpatialModel.lat_0
      ~TemplateSpatialModel.lon_0
      ~TemplateSpatialModel.map
      ~TemplateSpatialModel.map_center
      ~TemplateSpatialModel.parameters
      ~TemplateSpatialModel.parameters_unique_names
      ~TemplateSpatialModel.phi_0
      ~TemplateSpatialModel.position
      ~TemplateSpatialModel.position_error
      ~TemplateSpatialModel.position_lonlat
      ~TemplateSpatialModel.tag
      ~TemplateSpatialModel.type

   .. rubric:: Methods Summary

   .. autosummary::

      ~TemplateSpatialModel.__call__
      ~TemplateSpatialModel.copy
      ~TemplateSpatialModel.evaluate
      ~TemplateSpatialModel.evaluate_geom
      ~TemplateSpatialModel.freeze
      ~TemplateSpatialModel.from_dict
      ~TemplateSpatialModel.from_parameters
      ~TemplateSpatialModel.from_position
      ~TemplateSpatialModel.integrate_geom
      ~TemplateSpatialModel.plot
      ~TemplateSpatialModel.plot_error
      ~TemplateSpatialModel.plot_grid
      ~TemplateSpatialModel.plot_interactive
      ~TemplateSpatialModel.plot_position_error
      ~TemplateSpatialModel.read
      ~TemplateSpatialModel.reassign
      ~TemplateSpatialModel.to_dict
      ~TemplateSpatialModel.to_region
      ~TemplateSpatialModel.unfreeze
      ~TemplateSpatialModel.write

   .. rubric:: Attributes Documentation

   .. autoattribute:: covariance
   .. autoattribute:: default_parameters
   .. autoattribute:: evaluation_bin_size_min
   .. autoattribute:: evaluation_radius
   .. autoattribute:: evaluation_region
   .. autoattribute:: frozen
   .. autoattribute:: is_energy_dependent
   .. autoattribute:: lat_0
   .. autoattribute:: lon_0
   .. autoattribute:: map
   .. autoattribute:: map_center
   .. autoattribute:: parameters
   .. autoattribute:: parameters_unique_names
   .. autoattribute:: phi_0
   .. autoattribute:: position
   .. autoattribute:: position_error
   .. autoattribute:: position_lonlat
   .. autoattribute:: tag
   .. autoattribute:: type

   .. rubric:: Methods Documentation

   .. automethod:: __call__
   .. automethod:: copy
   .. automethod:: evaluate
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
   .. automethod:: read
   .. automethod:: reassign
   .. automethod:: to_dict
   .. automethod:: to_region
   .. automethod:: unfreeze
   .. automethod:: write
