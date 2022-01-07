# NeeDrop: Self-supervised Shape Representation from Sparse Point Clouds using Needle Dropping

<p align="center">
<img src="./doc/banner.gif" />
</p>

by: [Alexandre Boulch](https://www.boulch.eu), [Pierre-Alain Langlois](https://imagine.enpc.fr/~langloip/index.html), [Gilles Puy](https://sites.google.com/site/puygilles/) and [Renaud Marlet](http://imagine.enpc.fr/~marletr/)

[Project page](https://www.boulch.eu/2021_3dv_needrop) - Paper - [Arxiv](https://arxiv.org/abs/2111.15207) - Blog - [Code](https://github.com/valeoai/NeeDrop)

---
## Citation
```
@inproceedings{boulch2021needrop,
  title={NeeDrop: Self-supervised Shape Representation from Sparse Point Clouds using Needle Dropping},
  author={Boulch, Alexandre and Langlois, Pierre-Alain and Puy, Gilles and Marlet, Renaud},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}
```

---
## Dependencies


### Installation of generation modules

We use the generation code from [Occupancy Network](https://github.com/autonomousvision/convolutional_occupancy_networks). Please acknowledge the paper if you use this code.

```
python setup.py build_ext --inplace
```

---
## Datasets

### ShapeNet ([Occupancy Network](https://github.com/autonomousvision/convolutional_occupancy_networks) pre-processing)

We use the ShapeNet dataset as pre-processed by [Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks). Please refer to original repository for downloading the data.

### DFaust

Please download the data from the [official website](https://dfaust.is.tue.mpg.de/).

---
## Training
The following command trains the network with the default parameters. Here we assume the dataset is in a `data` folder. The outputs will be placed in a `results` folder.

### ShapeNet
```
python train.py --config configs/config_shapenet.yaml --log_mode interactive
```

### Finetuning with a reduced needle size
```
python train.py --config configs/config_shapenet.yaml --init_with results/ShapeNet_None_300_2048_filterNone/checkpoint.pth --sigma_multiplier 0.5 --experiment_name FT0.5 --lr_start 0.0001
```

---
## Generation

In order to generate the meshes, run the command

```
python generate.py --config replace/with/model/directory/config.yaml
```

If you want to generate a limited number of models per category:

```
python generate.py --config replace/with/model/directory/config.yaml --num_mesh 10
```

*Note*: evaluation on DFaust (as in [SAL](https://github.com/matanatz/SAL)) was done with 30000 points to compute the Chamfer distance.

```
python eval.py --dataset_name DFaust --dataset_root data/DFaust/ --prediction_dir results/DFaust_None_300_2048_filterNone/generation/ --n_points 30000
```


---
## Evaluation

To evaluate the model, run:

```
python eval.py --dataset_name ShapeNet --dataset_root data/ShapeNet/ --prediction_dir results/ShapeNet_None_300_2048_filterNone/generation/
```

---
## Pretrained models

### ShapeNet
| Model | IoU |
|---|---|
| [NeeDrop ShapeNet](https://github.com/valeoai/NeeDrop/releases/download/v0.0.0/ShapeNet_None_300_2048_filterNone.zip) | 0.663 |
| [NeeDrop ShapeNet + Finetuning 0.5](https://github.com/valeoai/NeeDrop/releases/download/v0.0.0/ShapeNet_FT0.5_300_2048_filterNone.zip) | 0.676 |

### DFaust
| Model | Chamfer 5% | Chamfer 50% | Chamfer 95% |
|---|---|---|---|
| [NeeDrop DFaust](https://github.com/valeoai/NeeDrop/releases/download/v0.0.0/DFaust_None_300_2048_filterNone.zip) | 0.269 * 1e-3 | 0.433 * 1e-3 | 1.149 * 1e-3 |
| [NeeDrop DFaust + Finetuning 0.5](https://github.com/valeoai/NeeDrop/releases/download/v0.0.0/DFaust_FT0.5_300_2048_filterNone.zip)| 0.107 * 1e-3 | 0.175 * 1e-3 | 1.322 * 1e-3 |

