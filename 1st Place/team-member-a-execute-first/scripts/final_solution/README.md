# Coldstart Energy Forecasting Final Solution

This folder contains all the scripts needed to reproduce the final solution
used on the challenge.

## Requirements

### Environment
This projects uses conda to create a python environment with all the dependencies needed
to train the models. Please install conda.

https://conda.io/docs/user-guide/install/index.html

After installing conda you can use the environment.yml file located at this folder. Please
do not use the one at the top folder because it won't be able to find all dependencies.

```bash
conda env create -f environment.yml
```

After creating the environment activate it and install seq2seq and recurrentshop libraries.
https://github.com/farizrahman4u/seq2seq


```bash
source activate coldstart
pip install git+https://www.github.com/farizrahman4u/recurrentshop.git
pip install git+https://github.com/farizrahman4u/seq2seq.git
```

Finally we have to install the coldstart library. Go to the upper level folder and run the following.

```bash
python setup.py develop
```

At this point we should have all the dependencies installed in our coldstart environment.

### Configuration
We need to set up a working folder. There we will store the data and the script will save also models
and submissions. The data needs a folder structure like the following.

```bash
guillermo@guillermo-MS-7998:/media/guillermo/Data/DrivenData/Cold_Start$ tree -L 2
├── data
│   ├── meta.csv
│   ├── submission_format.csv
│   ├── test.csv
│   └── train.csv
├── models
├── submissions
```

The models and submissions folders will be created automatically.


Once we have that structure we can go to coldstart/definitions.py and modify the DATASET_PATH to point
to our parent folder.

```python
DATASET_PATH = '/media/guillermo/Data/DrivenData/Cold_Start'
```

### GPU
The models can be trained on CPU, but it will take much longer. I recommend to use GPU for training.
In my case I used 2 GPUs that allowed to train up to 8 models in parallel.

## Script

To run the script we have to first activate the environment. The script will run for hours, it may take up to a day depending on the hardware.

```bash
source activate coldstart
./final_solution.sh
```

The main tasks that the script does are the following.

1. Adds more information to train and test files (weekday, is_day_off, is_holiday)
2. Finds the clusters in the data and creates the features that will be later be used on training
3. Train tailor made NN models. It will train for 5 folds using cross-validation.
4. Train seq2seq models. It will train for 5 folds using cross-validation.
5. Create invididual submissions
6. Merge submissions into final submission

At the end we will have something like this. The submission is final_solution.csv

```bash
.
├── data
│   ├── meta.csv
│   ├── submission_format.csv
│   ├── test.csv
│   └── train.csv
├── models
│   ├── seq2seq
│   └── tailor
└── submissions
    ├── final_solution.csv
    ├── seq2seq
    ├── seq2seq.csv
    ├── tailor
    └── tailor.csv
```
