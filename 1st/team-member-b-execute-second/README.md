drivendata-power-laws last_minute_team solution (twalen part)
=============================================================

This project contains scripts required to recreate our best team solution
for the contest:
[https://drivendata.org/competitions/55/schneider-cold-start/](https://drivendata.org/competitions/55/schneider-cold-start/)

System requirements
-------------------

I am getting consistent and deterministic results on following system:

* MacBook Pro, 8GB, macOS High Sierra 10.13
* macports (installed packages python36 and py36-virtualenvwrapper)

**Warning!** I tried to run this scripts on Google Cloud Linux machines,
it also runs deterministically but it generates slightly different results
for some models.

Extra requirements are installed inside virtualenv environment and include:

* pandas
* numpy
* scikit-learn
* keras
* tensorflow

External data (included in this repo):

* ironbar's best single submission (the scripts for recreating this solution are in separate repo)
* holidays (via Python package `holidays`)

Generating submission
---------------------

1. Copy the contest data to `data/raw` directory.
2. Run following commands:

```bash
make create_environment
workon drivendata-power-laws-solution
make gen
```

The recreated final submission is stored in `models/team-final-submission.csv`.

Details about generating solutions
----------------------------------

Running `make gen` generates following targets:

* `make requirements` - install required packages using `pip`
* `make data` - create interim and processed data:
  * `make data/interim/train_test.hdf5` - transform raw data to combined HDF5 file and add basic features
  * `make data/processed/train_test.hdf5` - generate final train test file
  * `make data/processed/selected-trivial-predictions.csv` - generate trivial predictions for selected five series
* `make models/twalen-0.2851.csv` - generate base twalen's submission
* `make models/twalen-20seeds.csv` - generate last twalen's prediction computed on 2018-10-31
* `make models/team-final-submission.csv` - combine `models/twalen-20seeds.csv` and `data/external/ironbar-best-submission-0.2881.csv` and `data/processed/selected-trivial-predictions.csv`

It takes about 2hrs to recreate the solution, the most time consuming part is 20seeds computation.

Solution writeup
------------------------------------

The brief description of the solution is in the `reports` directory.

Notebooks
------------------------------------

The EDA notebook is located in `notebooks` directory.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
