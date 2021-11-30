import numpy as np
import torch
import os
import glob
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
            [-0.5,-0.5,-0.5],
            [-0.5,-0.5, 0.5],
            [-0.5, 0.5,-0.5],
            [-0.5, 0.5, 0.5],
            [ 0.5,-0.5,-0.5],
            [ 0.5,-0.5, 0.5],
            [ 0.5, 0.5,-0.5],
            [ 0.5, 0.5, 0.5]
        ])


        print("Dataset -- getting filelists...", end="", flush=True)
        self.filelists = []
        if split in ["train", "training"]:
            for path in glob.glob(os.path.join(self.dataset_root,"*/train.lst")):
                self.filelists.append(path)
            for path in glob.glob(os.path.join(self.dataset_root,"*/val.lst")):
                self.filelists.append(path)
        elif split in ["test", "testing"]:
            for path in glob.glob(os.path.join(self.dataset_root,"*/test.lst")):
                self.filelists.append(path)

        self.filelists.sort()
        print("Done -",len(self.filelists), "files")

        print("Dataset -- getting the filenames...", end="", flush=True)
        self.filenames = []
        for flist in self.filelists:
            with open(flist) as f:
                dirname = os.path.dirname(flist)
                content = f.readlines()
                content = [line.split("\n")[0] for line in content]
                content = [os.path.join(dirname, line) for line in content]
                if num_mesh is not None:
                    content = content[:num_mesh]
            self.filenames += content
        print("Done -", len(self.filenames), "files")

        if self.filter_name is not None:
            fname_list = []
            for fname in self.filenames:
                if self.filter_name in fname:
                    fname_list.append(fname)
            self.filenames = fname_list
            print("Done -", len(self.filenames), "files")

        self.normalize_point_cloud = False

        if "shape_id" in kwargs and self.split=="test":
            self.shape_id = kwargs["shape_id"]
            self.iter_nbr = config["iter_test"]
            self.filenames = [self.filenames[self.shape_id]]
            self.points_shape = np.load(os.path.join(self.filenames[0], "pointcloud.npz"))["points"]
        else:
            self.points_shape = None
            self.shape_id = None

        self.metadata = {
            "04256520": {
                "id": "04256520",
                "name": "sofa",
                "class": 0
            },
            "02691156": {
                "id": "02691156",
                "name": "airplane",
                "class": 1
            },
            "03636649": {
                "id": "03636649",
                "name": "lamp",
                "class": 2
            },
            "04401088": {
                "id": "04401088",
                "name": "phone",
                "class": 3
            },
            "04530566": {
                "id": "04530566",
                "name": "vessel",
                "class": 4
            },
            "03691459": {
                "id": "03691459",
                "name": "speaker",
                "class": 5
            },
            "03001627": {
                "id": "03001627",
                "name": "chair",
                "class": 6
            },
            "02933112": {
                "id": "02933112",
                "name": "cabinet",
                "class": 7
            },
            "04379243": {
                "id": "04379243",
                "name": "table",
                "class": 8
            },
            "03211117": {
                "id": "03211117",
                "name": "display",
                "class": 9
            },
            "02958343": {
                "id": "02958343",
                "name": "car",
                "class": 10
            },
            "02828884": {
                "id": "02828884",
                "name": "bench",
                "class": 11
            },
            "04090263": {
                "id": "04090263",
                "name": "rifle",
                "class": 12
            }
        }

    def __len__(self):
        return len(self.filenames)

    def random_noise(self, points, stddev):
        noise = stddev * torch.randn_like(points)
        return points.clone() + noise
    
    def knn(self, pointcloud, queries, K):
        # with sklearn knn
        # neigh = NearestNeighbors(n_neighbors=K)
        # neigh.fit(pointcloud.numpy())
        # indices = neigh.kneighbors(queries.numpy(), K, return_distance=False)
        # indices = torch.tensor(indices, dtype=torch.long)
        # return indices
        tree = KDTree(pointcloud.numpy())
        distances, indices = tree.query(queries.numpy(), K)
        indices = torch.tensor(indices, dtype=torch.long)
        distances = torch.tensor(distances, dtype=torch.float)
        return distances, indices

    def get_save_filename_mesh(self, index):
        path = self.filenames[index].split('/')[-2:]
        path = os.path.join(*path)
        path = path + ".ply"
        return path

    def get_save_filename_pts(self, index):
        path = self.filenames[index].split('/')[-2:]
        path = os.path.join(*path)
        path = path + ".xyz"
        return path



    def get_manifold_points(self, index, num_manifold_points):
        filename = self.filenames[index]
        if self.points_shape is None:
            points_shape = np.load(os.path.join(filename, "pointcloud.npz"))["points"]
        else:
            points_shape = self.points_shape
        pts_shp = torch.tensor(points_shape, dtype=torch.float)
        if self.split == "test": # take always the same points
            pts_shp = pts_shp[:num_manifold_points]
        else:
            # pts_shp = pts_shp[torch.randperm(pts_shp.shape[0])[:num_manifold_points]]
            pts_shp = pts_shp[(torch.rand(num_manifold_points)*pts_shp.shape[0]).long()]
        return pts_shp


    def get_non_manifold_points(self, index, num_non_manifold_points, manifold_points):

        # get the points non manifold
        filename = self.filenames[index]
        data = np.load(os.path.join(filename, "points.npz"))
        points_space = data["points"]
        points_space = torch.tensor(points_space, dtype=torch.float)


        # sample more near the shape
        n_pts_sub = min(manifold_points.shape[0], num_non_manifold_points//4)
        num_non_manifold_rand = num_non_manifold_points - 3*n_pts_sub 

        non_manifold_rand_ids = torch.randperm(points_space.shape[0])[:num_non_manifold_rand]
        # non_manifold_rand_ids = (torch.rand(num_non_manifold_rand)*points_space.shape[0]).long()

        non_manifold_points = torch.cat([
                    manifold_points[:n_pts_sub] + torch.randn((n_pts_sub,3)) * 0.01,
                    manifold_points[:n_pts_sub] + torch.randn((n_pts_sub,3)) * 0.03,
                    manifold_points[:n_pts_sub] + torch.randn((n_pts_sub,3)) * 0.05,
                    points_space[non_manifold_rand_ids],
                    torch.tensor(self.box_points, dtype=torch.float)
        ], axis=0)


        # occupancies are available (used only to display metrics)
        occupancies = np.unpackbits(data["occupancies"])
        occupancies = occupancies[non_manifold_rand_ids]
        occupancies = torch.tensor(occupancies, dtype=torch.long)
        
        return non_manifold_points, occupancies


    def __getitem__(self, index):

        # get the manifold pts
        manifold_points = self.get_manifold_points(index, self.num_manifold_points)

        if self.cfg["pointcloud_noise"] > 0:
            manifold_points = self.random_noise(manifold_points, self.cfg["pointcloud_noise"])

        # get the non_manifold_points
        non_manifold_points, occupancies = self.get_non_manifold_points(index, self.num_non_manifold_points, manifold_points)

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
            "non_manifold_gt": occupancies
        }

        return return_dict