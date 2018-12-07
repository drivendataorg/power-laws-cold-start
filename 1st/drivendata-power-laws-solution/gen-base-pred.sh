#!/bin/bash
make data/processed/train_test.hdf5
LOG_DIR="models/prediction_twalen_0.2851"
INP=data/processed/train_test.hdf5

for w in hourly daily weekly ; do
    for cs in `seq 1 7` ; do
        python src/models/train_model.py --save-model \
        --log-dir=${LOG_DIR} --output=${LOG_DIR}/submission-${w}-${cs}.csv \
        --prediction-window=$w --cold-start-days=$cs --check-submission ${INP}
    done
done
./src/submission_tool.py \
    --input-dir=${LOG_DIR} \
    --patch-file data/processed/selected-trivial-predictions.csv \
    -o models/twalen-0.2851.csv

if [ -d expected-results ] ; then
    ./src/compare_submissions.py models/twalen-0.2851.csv expected-results/twalen-0.2851.csv
fi