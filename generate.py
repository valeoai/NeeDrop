from torch import optim
from trimesh.caching import DataStore
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

import utils.argparseFromFile as argparse
from utils.utils import wblue, wgreen
from network import Network

import yaml
import trimesh

import datasets

from utils.generator import Generator3D

from scipy.spatial import cKDTree as KDTree

from torch import distributions as dist

import pandas as pd
import logging

print_no_end= lambda s: print(s, end="", flush=True)

def invert_prediction_sign(points, latent, net):
    # predict an invertion of sign (only to orient the mesh)
    outputs_box = net.predict_from_latent(latent, points)
    invert_sign = (outputs_box[:,-8:].sum().item()>0)
    return invert_sign

def main(config):


    config["log_mode"] = "interactive"
    config = eval(str(config))
    config["mode"]="test" # for shapenet dataset TODO change that

    logging.info(config)

    # set the device
    device = torch.device(config['device'])

    # create the network
    logging.info("Creating the network")
    net = Network(latent_size=config["latent_size"], VAE=config["vae"])
    net.load_state_dict(torch.load(os.path.join(config["save_dir"], "checkpoint.pth"))["network"])
    net.to(device)
    net.eval()

    # create the dataset and dataloader
    logging.info("Creating the dataset")
    dataset = getattr(datasets, config["dataset_name"])(config, split="test", num_mesh=config["num_mesh"])

    logging.info("Creating the dataloader")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    # create the generator
    gen = Generator3D(net, points_batch_size=10000, 
            refinement_step=0,upsampling_steps=config["upsampling_steps"],
            threshold=0, resolution0=config["initial_voxel_size"], device=device)
    
    # result directories
    results_dir = os.path.join(config['save_dir'], config["generation_dir"])


    # compute mean and average
    t = tqdm(data_loader, ncols=110, disable=False)
    with torch.no_grad():

        for data_id, data in enumerate(t):

            pts_shape = data['manifold_points'].to(device)
            latent = net.get_latent(pts_shape)
            mesh_pred = gen.generate_mesh(latent)

            mesh_path = os.path.join(results_dir, "meshes", dataset.get_save_filename_mesh(data_id))
            pts_path = os.path.join(results_dir, "input", dataset.get_save_filename_pts(data_id))

            os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
            os.makedirs(os.path.dirname(pts_path), exist_ok=True)
            mesh_pred.export(mesh_path)
            np.savetxt(pts_path, data['manifold_points'][0].cpu().numpy())
            
        
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("trimesh").setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParserFromFile(description='Process some integers.')
    parser.add_argument('--config_default', type=str, default="configs/config_default.yaml")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_mesh', type=int, default=None)

    parser.update_file_arg_names(["config_default", "config"])
    config = parser.parse(use_unknown=True)
    
    logging.getLogger().setLevel(config["logging"])
    if config["logging"] == "DEBUG":
        config["threads"] = 0
    
    config["save_dir"] = os.path.dirname(config["config"])

    main(config)

