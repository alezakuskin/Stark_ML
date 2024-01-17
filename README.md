# StarkML: Machine Learning-based Tool for Predicting Stark Parameters of Atomic Emission Lines

## How to use

### Get predictions for your lines of interest

1. Download `Example.xlsx` and fill the table with your lines of interest. Express Energy ("*E lower*" and "*E upper*") in $cm^{-1}$ and Temperature in Kelvin. Fields 'Element', 'Wavelength' and 'Z number' are optional and used for information purposes only.

2. Download `Predictions.ipynb` notebook. You can run it locally on your machine, but we recommend opening it in Coogle Colaboratory. Link to open the notebook in Colaboratory appears on top of the content if you press on the name of notebook.

3. Follow instructions inside the notebook

### Reproduce results published in Zakuskin & Labutin, 2024, MNRAS, 527, 2

1. Run `Main_body.ipynb` notebook either locally or in Google Colaboratory. Follow instructions and comments inside the notebook. You will be able to choose models of interest, vary their hyperpapameters and enable/disable target scaling and data augmentation.

2. Run `Figures.ipynb` notebook either locally or in Google Colaboratory. It reproduces all figures from the paper and supplementary materials. **Note**, that you may want to switch values of `extended_train_set` and `scaled_target` variables between `True` and `False` depending on your needs.  

## Contents

Description of files and folders present in the depository

- `Example.xlsx`. Contains example of how to fill data on lines of your interest to get predictions on them.

- `Predictions.ipynb`. Interactive notebook that allows to predict Stark parameters for lines of your interest.

- `Main_body.ipynb`. Interactive notebook that reproduces all results published in *link* paper.

- `Figures.ipynb`. Interactive notebook that reproduces all visualizations present in paper *link* and supplementary materials to it.

- `Results` folder. Stores optimal hyperparameters for each combination of model and dataset in form of Python `dict`. Name of each file (e.g. `XGB_Extended_optimal_parameters`) contains model's name `XGB` and an indication of the dataset variant it was trained on (`Scaled` for *scaled*, `Extended` for *augmented*, `SHIFT` for Stark shift prediction or no specific indication for the raw dataset). You can find full table of optimal hyperparameters in Supplementary materials *link* (Table S1).

- `Source_files` folder. Contains full dataset used for training.

- `models` and `utils` folters. Contain `.py` files with implementation of model classes, functions for gyperparameters tuning, plotting the results, conversion terms to quantum numbers, train-test split and evaluation of models; performance.

## Citation

If you use predictions by our models, please cite our work:
```bib
@article{zakuskin2023StarkML,
 author={Zakuskin, Aleksandr and Labutin, Timur A.},
  journal={Monthly Notices of the Royal Astronomical Society}, 
  title={StarkML: application of machine learning to overcome lack of data on electron-impact broadening parameters}, 
  year={2024},
  volume={527},
  number={2},
  pages={3139--3145},
  doi={10.1093/mnras/stad3387}
}
```
