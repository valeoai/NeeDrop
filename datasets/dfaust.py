import torch
import os
from tqdm import *
import json
import trimesh
import numpy as np
from scipy.spatial import KDTree

class Dataset:

    def __init__(self, config, split="training", num_mesh=None, **kwargs):
        

        self.cfg = config

        self.dataset_root = config["dataset_root"]

        self.num_manifold_points = self.cfg["manifold_points"] if "manifold_points" in self.cfg else 1
        self.num_non_manifold_points = self.cfg["non_manifold_points"] if "non_manifold_points" in self.cfg else 1
        self.sigma_multiplier = self.cfg["sigma_multiplier"] if "sigma_multiplier" in self.cfg else 1
        self.filter_name = config["filter_name"] if "filter_name" in self.cfg else None

        self.split = split

        self.box_points = np.array([
            [-0.6,-0.6,-0.6],
            [-0.6,-0.6, 0.6],
            [-0.6, 0.6,-0.6],
            [-0.6, 0.6, 0.6],
            [ 0.6,-0.6,-0.6],
            [ 0.6,-0.6, 0.6],
            [ 0.6, 0.6,-0.6],
            [ 0.6, 0.6, 0.6]
        ])

        # normalization parameters
        self.translation = torch.tensor([0,-0.2288,0.1], dtype=torch.float)
        self.scale = 0.5

        # get the split
        if split in ["train", "training"]:
            with open("datasets/dfaust_splits/train_all_every5.json", "r") as f:
                filenames = json.load(f)
        elif split=="test":
            with open("datasets/dfaust_splits/test_all_every5.json", "r") as f:
                filenames = json.load(f)
        else:
            raise Exception("DFaust dataloader -- unknown")

        self.filelist = []

        for human,scans in filenames['scans'].items():
            for pose,shapes in scans.items():

                source = os.path.join(self.dataset_root, 'scans', human, pose)
                for shape in shapes:
                    self.filelist.append(os.path.join(source,shape) + '.ply')

        print("filelist --", len(self.filelist))

    def __len__(self):
        return len(self.filelist)
    
    def knn(self, pointcloud, queries, K):
        tree = KDTree(pointcloud.numpy())
        distances, indices = tree.query(queries.numpy(), K)
        indices = torch.tensor(indices, dtype=torch.long)
        distances = torch.tensor(distances, dtype=torch.float)
        return distances, indices

    def get_save_filename_mesh(self, index):
        path = "/".join(self.filelist[index].split("/")[-3:])
        return str(path)

    def get_save_filename_pts(self, index):
        path = "/".join(self.filelist[index].split("/")[-3:])
        path = str(path.replace(".ply", ".xyz"))
        return path

    def get_manifold_points(self, index, num_manifold_points):
        mesh = trimesh.load(self.filelist[index])
        # points_shape = trimesh.sample.sample_surface(mesh, num_manifold_points)[0]
        points_shape = mesh.vertices
        points_shape = points_shape.astype(np.float32)
        pts_shp = torch.tensor(points_shape, dtype=torch.float)
        pts_shp = pts_shp[(torch.rand(num_manifold_points)*pts_shp.shape[0]).long()]
        
        # fixed normalization
        pts_shp += self.translation
        pts_shp *= self.scale

        return pts_shp


    def get_non_manifold_points(self, index, num_non_manifold_points, manifold_points):

        num_manifold_points = min(manifold_points.shape[0], num_non_manifold_points//4)
        num_non_manifold_rand = num_non_manifold_points - 3*num_manifold_points

        non_manifold_points = torch.cat([
            manifold_points[:num_manifold_points] + torch.randn((num_manifold_points,3)) *0.01,
            manifold_points[:num_manifold_points] + torch.randn((num_manifold_points,3)) *0.03,
            manifold_points[:num_manifold_points] + torch.randn((num_manifold_points,3)) *0.05,
            torch.rand((num_non_manifold_rand, 3)) * 1.2 - 0.6,
            torch.tensor(self.box_points, dtype=torch.float)
        ])

        return non_manifold_points



    def __getitem__(self, index):

        # get the manifold pts
        manifold_points = self.get_manifold_points(index, self.num_manifold_points)

        if self.cfg["pointcloud_noise"] > 0:
            manifold_points = self.random_noise(manifold_points, self.cfg["pointcloud_noise"])

        # get the non_manifold_points
        non_manifold_points = self.get_non_manifold_points(index, self.num_non_manifold_points, manifold_points)

        # compute sigma
        distances, indices = self.knn(manifold_points, manifold_points, 2)
        sigma = distances.max(dim=1)[0] / 3
        sigma = sigma.unsqueeze(1).expand_as(manifold_points)
        if self.sigma_multiplier is not None:
            sigma = sigma * self.sigma_multiplier

        # compute the random perturbation
        h = torch.normal(torch.zeros_like(manifold_points), sigma)
        non_manifold_points_h1 = manifold_points+h
        non_manifold_points_h2 = manifold_points-h
        non_manifold_points_all = torch.cat([non_manifold_points_h1, 
            non_manifold_points_h2, 
            non_manifold_points], dim=0)

        # Points to points distanes
        distances, ids_pts2pts = self.knn(non_manifold_points_all, non_manifold_points, 2)
        max_ids = distances.max(dim=1)[1]
        max_ids = max_ids.unsqueeze(1)
        ids_pts2pts = ids_pts2pts.gather(1, max_ids).squeeze(0).squeeze(1)

        return_dict = {
            "shape_id": index,
            "manifold_points": manifold_points,
            "non_manifold_points": non_manifold_points_all,
            "ids_pts2pts": ids_pts2pts,
        }

        return return_dict


    def get_evaluation_material(self, index):

        eval_material = {}
        eval_material["mesh_gt"] = trimesh.load(self.filelist[index])

        return eval_material

    def get_category(self, f_id):
        return self.filelist[f_id].split("/")[-3]

    def get_object_name(self, f_id):
        return self.filelist[f_id].split("/")[-1]

    def get_class_name(self, f_id):
        return self.filelist[f_id].split("/")[-3]

    def unnormalizing_parameters(self, vertices):
        vertices /= self.scale
        vertices -= self.translation.numpy()
        return vertices
        