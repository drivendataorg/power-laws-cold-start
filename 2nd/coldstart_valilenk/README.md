coldstart_valilenk
==============================

Project with the second place model in Drivendata competition *Power Laws: Cold Start Energy Forecasting*

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    └── src                <- Source code for use in this project.
        ├── build_features.py <- Script for data processing and feature engineering
        └── model.py          <- Script for training model and making the prediction


--------
## Prerequisites:
Linux, Python 3.7, virtual environment according to requirements.txt

Manual setup:
```
conda create -n fastai_coldstart
conda activate fastai_coldstart
conda install -c pytorch pytorch-nightly cuda92
conda install -c fastai torchvision-nightly
conda install -c fastai fastai==1.0.19
conda install -c conda-forge pathos==0.2.1
conda install -c anaconda scikit-learn==0.20.0
```

## Hardware:
GPU required.

Original submit was made with 128Gb of RAM, 11GB Nvidia 1080Ti GPU.
If using less GPU RAM, you should make 'bs' (batch_size) and learning_rate (second parameter in learn.fit_one_cycle function) smaller. It can make the score worse and will train a bit slower.

## Running
0. Put original .csv files into 'data/raw' subfolder
1. Activate fastai_coldstart environment
2. Go to src folder
3. Run python build_features.py
4. Run python model.py

## Results
submit.csv file is in 'data' folder

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
