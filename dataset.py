from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import open3d as o3d
import os
import sys
sys.path.append(os.path.abspath("/home/arvc/Desktop/Antonio/minkowski/scripts/examples"))
from plyfile import PlyData
class preprocessingDataset2f(Dataset):
    """
    Rutas de interes:
    - "/media/arvc/Extreme SSD/kitti_mine/valid/[4-10]
    """

    def __init__(self,root_data, mode,voxel):
        super(Dataset, self).__init__()
        self.root = root_data
        self.directories = sorted(glob.glob('{}/*'.format(self.root)))
        self.pcds = []
        self.mode=mode
        self.voxel_size=voxel
        for i in self.directories:
            for j in glob.glob('{}/*'.format(i)):
                self.pcds.append(j)

    def __len__(self):
        # return len(self.pcds)
        return 1
    def __getitem__(self, idx):

        def normalize_features(features):
            norm_arr = np.empty_like(features)
            # for dim in range(features.shape[1]):
            minimo = min(features[:])
            diff_arr = max(features[:]) - minimo
            for n, l in enumerate(features[:]):
                norm_arr[n] = ((l - minimo) / diff_arr)
            return norm_arr.astype(np.float32)

        def compute_normals(pcd):
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.45, max_nn=30))
            pcd.orient_normals_to_align_with_direction()
            normals = np.asarray(pcd.normals)
            ey = o3d.geometry.PointCloud()
            ey.points = o3d.utility.Vector3dVector(pcd.points)
            ey.normals = o3d.utility.Vector3dVector(normals)
            return ey

        if self.mode=="train" or self.mode == "valid":
            pcd_raw = o3d.io.read_point_cloud(self.pcds[idx])
            self.coords = np.asarray(pcd_raw.points)
            self.coords = self.coords / self.voxel_size
            plydata = PlyData.read(self.pcds[idx])
            self.features = np.array((plydata.elements[0].data['z_normal'], plydata.elements[0].data['z'])).T
            self.label = np.array((plydata.elements[0].data['labels']))
            z_norm = normalize_features(self.features[:, 1])
            self.features[:, 1] = z_norm
            return self.coords, self.features, self.label
        else:
            pcd_raw = o3d.io.read_point_cloud(self.pcds[idx])
            self.coords_orig = np.asarray(pcd_raw.points)
            self.coords = self.coords_orig / self.voxel_size
            pcd = pcd_raw.voxel_down_sample_and_trace(0.03, pcd_raw.get_min_bound(), pcd_raw.get_max_bound(),approximate_class=True)
            cloud_with = compute_normals(pcd[0])
            final_normals = np.zeros((self.coords_orig.shape))
            for k, t in enumerate(pcd[2]):
                final_normals[np.asarray(t)] = np.asarray(cloud_with.normals)[k]
            self.features = np.stack((final_normals[:, 2], self.coords[:, 2]), axis=1)
            z_norm = normalize_features(self.features[:, 1])
            self.features[:, 1] = z_norm
            print(self.coords.shape, self.features.shape)
            return self.coords, self.features