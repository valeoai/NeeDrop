# NeeDrop: Self-supervised Shape Representation from Sparse Point Clouds using Needle Dropping

<p align="center">
<img src="./doc/banner.gif" />
</p>

by: [Alexandre Boulch](https://www.boulch.eu), [Pierre-Alain Langlois](https://imagine.enpc.fr/~langloip/index.html), [Gilles Puy](https://sites.google.com/site/puygilles/) and [Renaud Marlet](http://imagine.enpc.fr/~marletr/)

[Project page](https://www.boulch.eu/2021_3dv_needrop) - Paper - Arxiv - Blog - [Code](https://github.com/valeoai/NeeDrop)

## Dependencies


### Installation of generation modules

We use the generation code from [Occupancy Network](https://github.com/autonomousvision/convolutional_occupancy_networks). Please acknowledge the paper if you use this code.

```
python setup.py build_ext --inplace
```

## Datasets

### ShapeNet ([Occupancy Network](https://github.com/autonomousvision/convolutional_occupancy_networks) pre-processing)

We use the ShapeNet dataset as pre-processed by [Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks). Please refer to original repositiry for downloading the data.

## Training and evaluation

### ShapeNet

#### Training
The following command trains the network with the default parameters. Here we assume the dataset is in a `data` folder. The outputs will be placed in a `results` folder.

```
python train.py --config configs/config_shapenet.yaml --log_mode interactive
```

#### Generation

In order to generate the meshes, run the command

```
python generate.py --config replace/with/model/directory/config.yaml
```

If you want to generate a limited number of models per category:

```
python generate.py --config replace/with/model/directory/config.yaml --num_mesh 10
```