# Gaussian Processes on Finite Spaces of Graphs

(Preliminary version)

## Setup

This assumes conda is installed on your system \
If conda is not installed, download the [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html#)
If conda is installed, run the following commands:

```
./install_dependencies.sh
```

## Datasets
The datasets and splits are provided under `examples/data/freesolv`

## Splits
This step can be skipped if the splits are already present. To prepare splits, run the following steps:

```
python -m data_prep.splits --dataset freesolv --allowed_atoms C N O Cl --filename $FILENAME --split $SPLIT 
```
where `FILENAME` is the corresponding file in `examples/data/freesolv` and `SPLIT` is one of `{random, scaffold}`.
Note that in the experiments for the submission, we use the `random` split.

## Training

To train the model, run the following commands:
```
python -m scripts.train.run_molecule_gp --kernel $KERNEL --kernel_mode $MODE --lr 0.001 \
    --print_every 100 --eval_every 100 --train_iter 10000 \
    --kappa 1.0 --sigma2 1.0 --nu 2.5
```
where `KERNEL` is one of `{graph, projected (projected in the paper)}` and `MODE` is one of `{heat, matern}`


## Illustrations
Code used for generating illustrations can be found in `illustrations/`. 
Some additional packages might be necessary to download for these.