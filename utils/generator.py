import torch
import time
import trimesh
import numpy as np
import torch.optim as optim
from torch import autograd

from utils.libmise import MISE
from utils import libmcubes
from utils.libsimplify import simplify_mesh


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.
    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p

class Generator3D(object):
    '''  Generator class for Occupancy Networks.
    It provides functions to generate the final mesh as well refining options.
    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
        preprocessor (nn.Module): preprocessor for inputs
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 simplify_nfaces=None,
                 preprocessor=None,
                 invert_prediction_sign=False):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.preprocessor = preprocessor
        self.invert_prediction_sign = invert_prediction_sign

    def generate_mesh(self, latent, return_stats=True):
        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        mesh = self.generate_from_latent(latent)
        return mesh

    def generate_from_latent(self, latent):
        ''' Generates mesh from latent.
        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = self.threshold #np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Box
        pointsf = box_size * make_3d_grid(
            (-0.5,)*3, (0.5,)*3, (2,)*3
        )
        values = self.eval_points(pointsf, latent).cpu().numpy()
        value_grid = values.reshape(2, 2, 2)
        if value_grid.sum() > 0:
            self.invert_prediction_sign = True


        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            values = self.eval_points(pointsf, latent).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                values = self.eval_points(
                    pointsf, latent).cpu().numpy()
                values = values.astype(np.float64)

                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        # stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, latent)#z, c, stats_dict=stats_dict)
        return mesh

    def eval_points(self, p, latent):
        ''' Evaluates the occupancy values for the points.
        Args:
            p (tensor): points 
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.predict_from_latent(latent, pi, with_sigmoid=True)
                if self.invert_prediction_sign:
                    occ_hat *= -1

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def extract_mesh(self, occ_hat, latent):
        ''' Extracts the mesh from the predicted occupancy grid.
        Args:
            occ_hat (tensor): value grid of occupancies
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = self.threshold #np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()

        occ_hat_padded = occ_hat.copy()
        # occ_hat_padded = np.pad(
        #     occ_hat, 1, 'constant', constant_values=-1e6)

        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)

        # ##########
        # self.displacement_normal = vertices - np.floor(vertices * 10)/10
        # self.displacement_normal = (self.displacement_normal != 0).astype(float)
        # ##########


        # stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        # vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=None,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            # t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            # stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            # t0 = time.time()
            self.refine_mesh(mesh, occ_hat, latent)
            # stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, z, c=None):
        ''' Estimates the normals by computing the gradient of the objective.
        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        z, c = z.unsqueeze(0), c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, z, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals