SkyModel
========

.. currentmodule:: gammapy.modeling.models

.. autoclass:: SkyModel
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SkyModel.covariance
      ~SkyModel.default_parameters
      ~SkyModel.evaluation_bin_size_min
      ~SkyModel.evaluation_radius
      ~SkyModel.evaluation_region
      ~SkyModel.frame
      ~SkyModel.frozen
      ~SkyModel.name
      ~SkyModel.parameters
      ~SkyModel.parameters_unique_names
      ~SkyModel.position
      ~SkyModel.position_lonlat
      ~SkyModel.spatial_model
      ~SkyModel.spectral_model
      ~SkyModel.tag
      ~SkyModel.temporal_model
      ~SkyModel.type

   .. rubric:: Methods Summary

   .. autosummary::

      ~SkyModel.__call__
      ~SkyModel.contributes
      ~SkyModel.copy
      ~SkyModel.create
      ~SkyModel.evaluate
      ~SkyModel.evaluate_geom
      ~SkyModel.freeze
      ~SkyModel.from_dict
      ~SkyModel.from_parameters
      ~SkyModel.integrate_geom
      ~SkyModel.reassign
      ~SkyModel.to_dict
      ~SkyModel.unfreeze

   .. rubric:: Attributes Documentation

   .. autoattribute:: covariance
   .. autoattribute:: default_parameters
   .. autoattribute:: evaluation_bin_size_min
   .. autoattribute:: evaluation_radius
   .. autoattribute:: evaluation_region
   .. autoattribute:: frame
   .. autoattribute:: frozen
   .. autoattribute:: name
   .. autoattribute:: parameters
   .. autoattribute:: parameters_unique_names
   .. autoattribute:: position
   .. autoattribute:: position_lonlat
   .. autoattribute:: spatial_model
   .. autoattribute:: spectral_model
   .. autoattribute:: tag
   .. autoattribute:: temporal_model
   .. autoattribute:: type

   .. rubric:: Methods Documentation

   .. automethod:: __call__
   .. automethod:: contributes
   .. automethod:: copy
   .. automethod:: create
   .. automethod:: evaluate
   .. automethod:: evaluate_geom
   .. automethod:: freeze
   .. automethod:: from_dict
   .. automethod:: from_parameters
   .. automethod:: integrate_geom
   .. automethod:: reassign
   .. automethod:: to_dict
   .. automethod:: unfreeze
