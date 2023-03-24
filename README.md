# Gaussian Processes on Finite Spaces of Graphs

(Preliminary version)

## Updates:

[01.03.2023]: As a small correction, the column RMSE in the paper should be replaced with L2-Error. The qualitative results remain unchanged though.

## Credits

The code for this work was developed by [Vignesh Ram Somnath](https://github.com/vsomnath) and [Mohammad Reza Karimi](https://github.com/moreka). The development-time commit history was erased in a deadline rush while transferring the repository.

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
where `KERNEL` is one of `{graph, projected}` and `MODE` is one of `{heat, matern}`

## License

The project is listed under the MIT license. Please see [LICENSE](https://github.com/IBM/graph_space_gps/blob/main/LICENSE) for more details.

## Reference

If you find our code useful, please cite our paper:

```
@inproceedings{borovitskiy2023isotropic,
      title={Isotropic Gaussian Processes on Finite Spaces of Graphs}, 
      author={Borovitskiy, Viacheslav and Karimi, Mohammad Reza and Somnath, Vignesh Ram and Krause, Andreas},
      booktitle={International Conference on Artificial Intelligence and Statistics},
      year={2023},
      organization={PMLR}
}
```

## Contact

If you have any questions about our code, or want to report a bug, please raise a GitHub issue.
