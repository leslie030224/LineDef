#!/bin/bash

datasets=("activemq" "camel" "derby" "groovy" "hbase" "jruby" "lucene" "wicket" "hive")

for dataset in "${datasets[@]}"; do
   python generate_prediction.py -dataset "$dataset" -target_epochs 10
done

echo "done！"