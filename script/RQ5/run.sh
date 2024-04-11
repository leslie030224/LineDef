#!/bin/bash

# 数据集列表
datasets=("activemq" "camel" "derby" "groovy" "hbase" "jruby" "lucene" "wicket" "hive")

dirs=(
    "./single"
    "./weighted_single"
    "./weighted_double"
)

# 循环遍历每个数据集并执行训练命令

for dir in "${dirs[@]}"; do
    cd "$dir"
    for dataset in "${datasets[@]}"; do
        python train_model.py -dataset "$dataset" -lr 0.001 -num_epochs 15 -exp_name "${dir#./}"
    done

    for dataset in "${datasets[@]}"; do
        python generate_prediction.py -dataset "$dataset" -target_epochs 10 -exp_name "${dir#./}"
    done

    echo "当前目录$dir 中所有数据集训练已完成！"
    cd ..
done

Rscript get_evaluation_result_RQ5.R

echo "done!"
