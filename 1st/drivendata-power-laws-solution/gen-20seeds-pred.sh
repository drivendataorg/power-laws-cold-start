#!/bin/bash
LOG_DIR="models/prediction_20seeds"
INP=data/processed/train_test.hdf5

for rs in `seq 1 20` ; do
    for w in hourly daily weekly ; do
        for cs in 7 ; do
            python src/models/train_model.py --save-model \
                --log-dir=${LOG_DIR} \
                --output=${LOG_DIR}/submission-${w}-${cs}-rs${rs}.csv \
                --prediction-window=${w} --cold-start-days=${cs} --random-state=${rs} \
                ${INP}
            done
    done
done
./src/submission_tool.py \
    --input-dir=${LOG_DIR} \
    --patch-file data/processed/selected-trivial-predictions.csv \
    -o models/twalen-20seeds-cs7.csv || exit 1
./src/submission_tool.py \
    --input-file=models/twalen-0.2851.csv \
    --patch-file models/twalen-20seeds-cs7.csv \
    -o models/twalen-20seeds.csv || exit 1

if [ -d expected-results ] ; then
    ./src/compare_submissions.py models/twalen-20seeds.csv expected-results/twalen-20seeds.csv
fi