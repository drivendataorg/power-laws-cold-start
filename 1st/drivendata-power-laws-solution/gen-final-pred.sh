#!/bin/bash
./src/submission_tool.py \
    --input-file=models/twalen-20seeds.csv \
    --input-file=data/external/ironbar-best-submission-0.2881.csv \
    --patch-file data/processed/selected-trivial-predictions.csv \
    -o models/team-final-submission.csv
if [ -d expected-results ] ; then
    ./src/compare_submissions.py models/team-final-submission.csv expected-results/team-final-submission.csv
fi