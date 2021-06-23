# Notebooks

The Notebooks folder contains analysis and development of the project

### The code was structured following this order:

-notebooks

	-evaluating
        . autoML_robstuness_evaluation.ipynb - Evaluate AutoML model's robustness by training it and assessing it performance for different sizes of data sets.
        . autoML_sigma_MAD.ipynb - Evaluate AutoML model's by the metrics of sigma MAD and other metrics presented on https://arxiv.org/pdf/1806.06607.pdf .
        . regression_only-IA_gp.ipynb - Applies some data filtering cutting objects with few gaussian process fitting points for ordinary regression models. Attempt to improve model results in a data centric way.
        . regression_only-IA_outliers.ipynb - Applies some data filtering cutting outliers for ordinary regression models. Attempt to improve model results in a data centric way.

	-modelling
        . regression.ipynb - Applies traditional regression models to predict the redshift values.
        . regression_only-IA.ipynb - Applies traditional regression models to predict the redshift values only considering supernovae from type IA.
        . autoML.ipynb - Applies H2O AutoML models to predict the redshift values.
        . autoML_only-IA.ipynb - Applies H2O AutoML models to predict the redshift values only considering supernovae from type IA.
        . autoML_ensemble_predictions.ipynb - Uses the models from AutoML saved on MOJO format to create a gaussian KDE PDF from their predictions.

	-processing
        . gen_labels.ipynb - Read raw data and creates the labels for the supervised problem.
        . gaussian_process.ipynb - Reproduces the pipeline processing steps from Lochner (https://arxiv.org/pdf/1603.00882.pdf) and dos Santos (https://arxiv.org/pdf/1908.04210.pdf).