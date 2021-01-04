# Redshift Prediction

## Source code and notebooks used during Article ` Full Article Name Here `

Different data frames used on this project were saved on pickle files in order to promote a pipeline checkpoint.

## The code was structured following this order:
```
-data
	-enriched
	-raw
	-structured
	
-models

-notebooks
	-evaluating
	-modelling
	-processing
	
-src
	-modelling
	-precessing
``` 
- data: contain all pickles, .csv and .DAT used and generated during project.
    * data/enriched: contain results, files with evaluation comparing models and data frames with final models' predictions.
    * data/raw: contain files from LSST survey (download link here { ` PEGAR LINK COM O MARCELO ` })
    * data/structured: contain data frames and csv files used during the project development. Checkpoint like files to debug, continuation and evaluation.

- notebooks: folder containing analysis and development of the project.

    * notebooks/processing: first notebooks. Responsible for pipelines to generate features and target values.
    * notebooks/modeling: notebooks in which models were developed. 
    * notebooks/evaluating: notebooks in which models were evaluated to assess their efficiency and score.

- src: Python files with functions used during notebooks' procedures.

All models and data are available for downloading at the following link: https://drive.google.com/drive/folders/1FP3515CzNqQJY0PZFSWT9y7gC8ldi8pm?usp=sharing .

For proper notebook running they must ne unzipped on the root of this project. 


