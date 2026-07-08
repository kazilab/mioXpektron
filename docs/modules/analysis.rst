Analysis
========

The analysis module provides downstream statistical tools for aligned peak
matrices produced by the pipeline or :func:`~mioXpektron.utils.main.batch_processing`.

Quick Example
-------------

.. code-block:: python

   from mioXpektron import AnalysisConfig, AnalysisWorkflow

   config = AnalysisConfig(
       outdir="analysis_outputs",
       group_a="Treatment",
       group_b="Control",
       run_ml_benchmark=True,
       run_ml_tuning=True,
   )
   results = AnalysisWorkflow(intensity_df.reset_index(), config=config).run()

Matrix Preparation
------------------

.. autofunction:: mioXpektron.analysis.prepare_matrix

.. autofunction:: mioXpektron.analysis.infer_feature_columns

Univariate Statistics
---------------------

.. autofunction:: mioXpektron.analysis.bh_fdr

.. autofunction:: mioXpektron.analysis.compute_univariate_tests

Visualization
-------------

.. autofunction:: mioXpektron.analysis.plot_volcano

.. autofunction:: mioXpektron.analysis.plot_pca

.. autofunction:: mioXpektron.analysis.plot_umap

.. autofunction:: mioXpektron.analysis.plot_tsne

.. autofunction:: mioXpektron.analysis.run_embeddings

.. autofunction:: mioXpektron.analysis.plot_heatmap_top_features

Optional Dependencies
---------------------

Extended analysis features mirror the xpectrass stack. Install with::

   pip install mioXpektron[analysis]

This adds ``umap-learn``, ``xgboost``, and ``shap``. t-SNE and cNMF use
scikit-learn and are always available.

.. autofunction:: mioXpektron.analysis.analysis_capabilities

.. autofunction:: mioXpektron.analysis.missing_packages

Machine Learning
----------------

.. autofunction:: mioXpektron.analysis.prepare_ml_data

.. autofunction:: mioXpektron.analysis.get_benchmark_models

.. autofunction:: mioXpektron.analysis.evaluate_model

.. autofunction:: mioXpektron.analysis.evaluate_all_models

Classification Metrics
----------------------

.. autofunction:: mioXpektron.analysis.calculate_multiclass_metrics

.. autofunction:: mioXpektron.analysis.plot_confusion_matrix

Hyperparameter Tuning
---------------------

.. autofunction:: mioXpektron.analysis.tune_top_models

.. autofunction:: mioXpektron.analysis.get_tuning_grid

Multi-Dataset Comparison
------------------------

.. autofunction:: mioXpektron.analysis.compare_model_results

.. autofunction:: mioXpektron.analysis.run_multi_dataset_comparison

.. autofunction:: mioXpektron.analysis.summarize_model_families

Consensus NMF
-------------

.. autofunction:: mioXpektron.analysis.run_cnmf

.. autofunction:: mioXpektron.analysis.choose_k_by_pac

.. autofunction:: mioXpektron.analysis.plot_pac_vs_k

.. autofunction:: mioXpektron.analysis.explain_with_shap

Workflow
--------

.. autoclass:: mioXpektron.analysis.AnalysisConfig
   :members:

.. autoclass:: mioXpektron.analysis.AnalysisWorkflow
   :members: run

.. autofunction:: mioXpektron.analysis.run_analysis