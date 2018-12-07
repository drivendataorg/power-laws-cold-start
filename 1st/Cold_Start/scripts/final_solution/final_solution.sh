#!/bin/bash
python prepare_data.py
python find_clusters.py
python create_cluster_features.py
python tailor_train.py
python seq2seq_train.py
python tailor_submission.py
python seq2seq_submission.py
python combine_submissions.py