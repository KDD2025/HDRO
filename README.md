# HDRO

Official implementation of our paper [Robust Multi-scenario Recommendation with Hierarchical Distributed Robust Optimization]


## Introduction


## Dataset Download
In this paper, we use three datasets, **MovieLens**, **Douban**, **KuaiShou**.

## Run the model
```Shell
cd examples
# For multi-scenario model STAR on dataset MovieLens
python run_movielens_train_val_test_multi_scenario.py --data_name=ml-1m --model_name_type HDRO --model_name=STAR --epoch 100 --batch_size=512 --seed 2022 --gpu_id 2 --learning_rate_v=0.001 --learning_rate_w=1e-4 --learning_rate_c=1e-8 --learning_rate_f=1e-3 -alpha=0 -beta=0 -lambda_w=1e-5 --cluster_class=10 --pre_distribution=2 -tau1=0.8 -tau2=0.8

```





