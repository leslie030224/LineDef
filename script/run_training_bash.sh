#!/bin/bash

datasets=("activemq" "camel" "derby" "groovy" "hbase" "jruby" "lucene" "wicket" "hive")

for dataset in "${datasets[@]}"; do
    python train_model.py -dataset "$dataset" -lr 0.001 -num_epochs 20
done

echo "done！"