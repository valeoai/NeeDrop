from torch import optim
from trimesh.caching import DataStore
import os
from tqdm import tqdm
import numpy as np
import utils.argparseFromFile as argparse
from utils.utils import wblue, wgreen
import trimesh
import datasets
from scipy.spatial import cKDTree as KDTree
import logging

from utils.libmesh import check_mesh_contains
from utils.implicit_waterproofing import implicit_waterproofing

import pandas as pd


from multiprocessing import Pool
import itertools
from functools import partial
import time


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

def eval_mesh(mesh_pred, gt_data, n_points=100000):

    # predictions
    if (not isinstance(mesh_pred, trimesh.Scene)) and (len(mesh_pred.vertices) !=0) and (len(mesh_pred.faces)!=0):
        pointcloud_pred, idx = mesh_pred.sample(n_points, return_index=True)
        pointcloud_pred = pointcloud_pred.astype(np.float32)
        normals_pred = mesh_pred.face_normals[idx]
    else:
        pointcloud_pred = np.empty((0, 3))
        normals_pred = np.empty((0, 3))

    mesh_gt = gt_data.get("mesh_gt")
    if mesh_gt is not None:
        pointcloud_gt, idx = gt_data["mesh_gt"].sample(n_points, return_index=True)
        pointcloud_gt = pointcloud_gt.astype(np.float32)
        normals_gt = gt_data["mesh_gt"].face_normals[idx]
    else:
        pointcloud_gt = gt_data["pointcloud_gt"]
        normals_gt = gt_data["normals_gt"]
        idx = np.random.randint(pointcloud_gt.shape[0], size=n_points)
        pointcloud_gt = pointcloud_gt[idx]
        normals_gt = normals_gt[idx]

    # convert to float 32
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    pointcloud_gt = pointcloud_gt.astype(np.float32)
    normals_pred = normals_pred.astype(np.float32)
    normals_gt = normals_gt.astype(np.float32)

    out_dict = eval_pointcloud(pointcloud_pred, pointcloud_gt, normals_pred, normals_gt)

    points_iou = gt_data.get("points_iou")
    occ_gt = gt_data.get("occ_gt")
    if (points_iou is not None):

        if occ_gt is not None:
            if len(mesh_pred.vertices) != 0 and len(mesh_pred.faces) != 0:
                occ = check_mesh_contains(mesh_pred, points_iou)
                out_dict['iou'] = compute_iou(occ, occ_gt)
            else:
                out_dict['iou'] = 0.
        else:
            
            raise NotImplementedError

            # need test to be uncommented
            # bb_max = points_iou.max(axis=0)
            # bb_min = points_iou.min(axis=0)
            # bb_len = bb_max - bb_min
            # bb_samples = np.random.rand(n_points*10, 3) * bb_len + bb_min
            # occ_pred = implicit_waterproofing(mesh_pred, bb_samples)[0]
            # occ_gt = implicit_waterproofing(mesh_gt, bb_samples)[0]
            # area_union = (occ_pred | occ_gt).astype(np.float32).sum()
            # area_intersect = (occ_pred & occ_gt).astype(np.float32).sum()
            # out_dict['iou'] =  (area_intersect / area_union)

    return out_dict


def eval_pointcloud(pointcloud_pred, pointcloud_gt,
                    normals_pred=None, normals_gt=None):

    pointcloud_pred = np.asarray(pointcloud_pred)
    pointcloud_gt = np.asarray(pointcloud_gt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    completeness, completeness_normals = distance_p2p(
        pointcloud_gt, pointcloud_pred,
        normals_gt, normals_pred
    )
    completeness2 = completeness ** 2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()


    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, pointcloud_gt,
        normals_pred, normals_gt
    )
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()

    # Chamfer distance
    chamfer_l2 = 0.5 * completeness2 + 0.5 * accuracy2
    chamfer_l1 = 0.5 * completeness + 0.5 * accuracy

    if normals_pred is not None:
        accuracy_normals = accuracy_normals.mean()
        completeness_normals = completeness_normals.mean()
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
    else:
        accuracy_normals = np.nan
        completeness_normals = np.nan
        normals_correctness = np.nan


    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normals': normals_correctness,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer_l2': chamfer_l2,
        'chamfer_l1': chamfer_l1,
        'iou': np.nan
    }

    return out_dict


def distance_p2p(pointcloud_pred, pointcloud_gt,
                    normals_pred, normals_gt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(pointcloud_gt)
    dist, idx = kdtree.query(pointcloud_pred)

    if normals_pred is None:
        return dist, None

    normals_pred = normals_pred / np.linalg.norm(normals_pred, axis=-1, keepdims=True)
    normals_gt = normals_gt / np.linalg.norm(normals_gt, axis=-1, keepdims=True)

    normals_dot_product = (normals_gt[idx] * normals_pred).sum(axis=-1)
    # Handle normals that point into wrong direction gracefully
    # (mostly due to mehtod not caring about this in generation)
    normals_dot_product = np.abs(normals_dot_product)

    return dist, normals_dot_product


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_function(data_id, dataset, results_dir, n_points=100000):
    
    # get the predicted mesh
    mesh_path = os.path.join(results_dir, "meshes", dataset.get_save_filename_mesh(data_id))
    predicted_mesh = trimesh.load_mesh(mesh_path, force='mesh')

    if isinstance(predicted_mesh, trimesh.Scene):
        print("Scene error - no mesh was produced --> mesh replaced by a sphere ")
        predicted_mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5, color=None)

    # get the original data
    gt_data = dataset.get_evaluation_material(data_id)

    # evaluate the mesh
    out_dict = eval_mesh(predicted_mesh, gt_data)

    # print(out_dict["iou"])

    category = dataset.get_category(data_id)
    object_name = dataset.get_object_name(data_id)
    class_name = dataset.get_class_name(data_id)

    out_dict['idx'] = data_id
    out_dict['class'] = class_name
    out_dict['category'] = category
    out_dict['name'] = object_name

    return out_dict


def main(config):

    # create the dataset and dataloader
    logging.info("Creating the dataset")
    dataset = getattr(datasets, config["dataset_name"])(config, split="test", num_mesh=config["num_mesh"])
    
    # result directories
    results_dir = config["prediction_dir"]

    # compute mean and average
    eval_dicts = []
    ids = list(range(len(dataset)))
    for ch in tqdm(list(chunks(ids, config["threads"])), ncols=100):
        with Pool(config["threads"]) as p:
            chunk_eval_dicts = p.map(partial(process_function, dataset=dataset, results_dir=results_dir, n_points=config["n_points"]), ch)
            eval_dicts += chunk_eval_dicts

    # define the output files
    out_file = os.path.join(results_dir, 'eval_meshes_full.pkl')
    out_file_class = os.path.join(results_dir, 'eval_meshes.csv')
    out_file_quantile = os.path.join(results_dir, 'eval_meshes_quantile.csv')

    # Create pandas dataframe and save
    eval_df = pd.DataFrame(eval_dicts)
    eval_df.set_index(['idx'], inplace=True)
    eval_df.to_pickle(out_file)

    # Create CSV file  with main statistics
    eval_df_class = eval_df.groupby(by=['class']).mean()
    eval_df_class.loc['mean'] = eval_df_class.mean()
    eval_df_class.to_csv(out_file_class)
    print(eval_df_class)

    # quantiles
    eval_df.quantile([0.05,0.5,0.95]).to_csv(out_file_quantile)
            

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParserFromFile(description='Process some integers.')
    parser.add_argument('--dataset_name', type=str, default='ShapeNet')
    parser.add_argument('--dataset_root', type=str, default='data/ShapeNet')
    parser.add_argument('--prediction_dir', type=str, required=True)
    parser.add_argument('--num_mesh', type=int, default=None)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--n_points', type=int, default=100000)
    config = parser.parse(use_unknown=True)

    main(config)

