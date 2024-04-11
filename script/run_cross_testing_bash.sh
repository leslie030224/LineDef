#!/bin/bash

datasets=("activemq" "camel" "derby" "groovy" "hbase" "jruby" "lucene" "wicket" "hive")

for dataset in "${datasets[@]}"; do
   python generate_prediction_cross_projects.py -dataset "$dataset"  -target_epochs 10
done

echo "done！"