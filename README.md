# ACE: An Attention-based Model for Cardinality Estimation of Set-Valued Queries
## Introduction
This repo contains the source code of ACE

## Quick Start
The link of three datasets with generated query workloads can be downloaded via [this link](https://drive.google.com/drive/folders/15jcyAAGUkAx30qHAbHz5aWa9jzy0iMij?usp=sharing). Running the code mainly includes 2 steps: (1) **Featurization** represents the underlying data numerically and outputs the distilled matrix. (2) **Estimation** utilizes the data matrix and the queried element embeddings to predicate the cardinality. 

*For the e2e experiment, please refer to [this repo](https://github.com/dbis-ukon/lplm)*.

### Before running
You need to change the folder path. You can search `TODO` to see all places.

### Featurization (offline)
To train a model to represent data, you could run:
```bash
python3 data_encoder.py --d [dataset] --r [distill ratio] --dis_dep [distill depth]
```
This command will split the dataset into training and testing data. Then the trained aggregation and distillation models are stored in the folder `./save_model`.

Then, you could run the following command to generate the distilled dataset representation.
```bash
python3 data_representation.py --d [dataset] --r [distill ratio] --dis_sep [distill depth]
```

The generated representation is also stored in the folder `./save_model`.

### Estimation (online)
After generating the representation, you could train the estimator by running:
```bash
python3 query_analyzer.py --d [dataset] --qt [query type]
```

Additionally, you could use the trained model to get the estimation by running:
```bash
python3 query_analyzer.py --d [dataset] --m test --qt [query type] --qf [frequency]
```

Here, the query workloads are divided into three classes based on their type: **superset**, **subset** and **overlap**. Additionally, each query workload are also partitioned into three sub-classes based on the frequency of its comprised elements: **regular**(considering all elements), **high**(only considering high-frequency element) and **low**(only considering high-frequency element).

## Dynamic Data

The whole process is similar to the static one.

You first need to train the encoder by using the `data_encoder.py` file. Then, you can use the `dynamic/dy_data_representation.py` file to generate the representation of each update epoch. Finally, you need to run `dynamic/dy_query_analyzer.py` to see the update time (train) and the Q-error (test).

## Ablation Study

Except the CA ablation study, it is easy to modify the original code. For example, for the AG ablation study, you only need to exclude the feed-forward network. Additionally, we have commented (`SA ablation`) the parts that you need to use in the SA ablation study.

In terms of the CA ablation study, you need to change two files: `query_analyzer.py` and `model/ace.py`. In these files, you need to uncomment the part with `CA ablation` comment and comment the `original ACE` part.


