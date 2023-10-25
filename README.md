# StarkML: Machine Learning-based Tool for Predicting Stark Parameters of Atomic Emission Lines

## How to use

### Get predictions for your lines of interest

1. Download `Example.xlsx` and fill the table with your lines of interest. Express Energy ("*E lower*" and "*E upper*") in $cm^{-1}$ and Temperature in Kelvin. Fields 'Element', 'Wavelength' and 'Z number' are optional and used for information purposes only.

2. Download `Predictions.ipynb` notebook. You can run it locally on your machine, but we recommend opening it in Coogle Colaboratory. Link to open the notebook in Colaboratory appears on top of the content if you press on the name of notebook.

3. Follow instructions inside the notebook

### Reproduce results published in MNRAS <link will be available later>

1. Run `Main_body.ipynb` notebook either locally or in Google Colaboratory. Follow instructions and comments inside the notebook. You will be able to choose models of interest, vary their hyperpapameters and enable/disable target scaling and data augmentation.

2. Run `Figures.ipynb` notebook either locally or in Google Colaboratory. It reproduces all figures from the paper and supplementary materials. **Note**, that you may want to switch values of `extended_train_set` and `scaled_target` variables between `True` and `False` depending on your needs.  

## Contents

In the present repository you find everything needed to predict Stark broadening parameters for atomic lines of your interest. Currenly, predictions are made by an optimized XGBoost model.

### Please, follow these simple steps to get predicted values:
