# OptFeature

This repository contains the official implementation of our NeurIPS 2023 paper: 
  - Towards Hybrid-grained Feature Interaction Selection for Deep Sparse Network.

Kindly view our preprint [here](https://arxiv.org/abs/2310.15342).

### Data Preprocessing

You can prepare the Criteo data in the following format. Avazu and KDD12 datasets can be preprocessed by calling its own python file.

```
python datatransform/criteo2tf.py --store_stat --stats PATH_TO_STORE_STATS
		--dataset RAW_DATASET_FILE --record PATH_TO_PROCESSED_DATASET \
		--threshold YOUR_THRESHOLD --ratio 0.8 0.1 0.1 \
```

Then you can find a your processed files in the tfrecord format under the `PATH_TO_PROCESSED_DATASET` folder. You may also need to record the number of feature and field respectively.

### Run

To run Baselines:
```
python trainer.py $DATASET $MODEL \
        --feature $NUM_OF_FEATURE --field $NUM_OF_FIELD \
        --data_dir $PATH_TO_PROCESSED_DATASET
```

To run OptFeature:
```
python searcher.py $DATASET $MODEL \
        --feature $NUM_OF_FEATURE --field $NUM_OF_FIELD \
        --data_dir $DATA_DIR
```

### Hyperparameter Settings

Here we list the hyper-parameters we used in the following table. Note that due to the dataset partition, the optimal hyper-parameters over your dataset may differ.

| Dataset | LR | L2 | FI_LR |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Criteo | 3e-4 | 3e-6 | 3e-5 | 
| Avazu  | 3e-4 | 3e-6 | 3e-5 |
| KDD12  | 3e-5 | 1e-5 | 3e-5 |
