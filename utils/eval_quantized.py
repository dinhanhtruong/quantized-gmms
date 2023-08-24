import point_cloud_utils as pcu
import numpy as np
import os
from tqdm import tqdm

# contains continuous and quantized output mesh OBJs
exp_name = 'zb_quantize'
mesh_dir = f'assets/checkpoints/spaghetti_airplanes/{exp_name}/occ' # relative to root

# get predicted and 'gt' mesh filepaths
all_meshes = os.listdir(mesh_dir)
num_comparisons = len(all_meshes)//2
print("num comparisons: ", num_comparisons)
gt_filepaths = [f'{mesh_dir}/{file}' for file in all_meshes if not file.endswith("_quantized.obj")]
predicted_filepaths = [f'{mesh_dir}/{file}' for file in all_meshes if file.endswith("_quantized.obj")]
assert len(gt_filepaths) == len(predicted_filepaths)

# compute chamfer + EMD
chamfer_dists = []
EMDs = []
for gt_file, pred_file in tqdm(zip(gt_filepaths, predicted_filepaths), total=num_comparisons):
    # sample uniform points on surface
    n_samples = 30000 # for chamfer, same as SPAGHETTI
    n_samples_emd = 768
    # gt cloud
    gt_v, gt_f, gt_n = pcu.load_mesh_vfn(gt_file)
    face_ids, bary_coords = pcu.sample_mesh_random(gt_v, gt_f, n_samples)
    gt_samples = pcu.interpolate_barycentric_coords(gt_f, face_ids, bary_coords, gt_v)

    # predicted cloud
    pred_v, pred_f, pred_n = pcu.load_mesh_vfn(pred_file)
    face_ids, bary_coords = pcu.sample_mesh_random(pred_v, pred_f, n_samples)
    pred_samples = pcu.interpolate_barycentric_coords(pred_f, face_ids, bary_coords, pred_v)

    chamfer_dists.append(pcu.chamfer_distance(gt_samples, pred_samples))
    EMDs.append(pcu.earth_movers_distance(gt_samples[:n_samples_emd], pred_samples[:n_samples_emd])[0]) # ignore pairwise matrix

chamfer_dists = np.array(chamfer_dists)
EMDs = np.array(EMDs)
print(f"Chamfer: \n \t mean {np.mean(chamfer_dists)} \t median {np.median(chamfer_dists)}")
print(f"EMD: \n \t mean {np.mean(EMDs)} \t median {np.median(EMDs)}")