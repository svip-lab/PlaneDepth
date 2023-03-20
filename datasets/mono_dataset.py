# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from ast import Try

from PIL import Image  # using pillow-simd for increased speed
import os,  subprocess
import random
from matplotlib import scale
import numpy as np
import copy, shutil, json
#from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F

from utils import *
from . import pair_transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 novel_frame_ids,
                 is_train=False,
                 use_crop=True,
                 colmap_path="./kitti_colmap",
                 use_colmap=True,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.interp = Image.ANTIALIAS

        self.novel_frame_ids = novel_frame_ids

        self.is_train = is_train
        self.use_crop = use_crop
        self.use_colmap = use_colmap and self.is_train
        self.colmap_path = colmap_path
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = pair_transforms.ToTensor()
        if self.use_crop:
            self.data_aug = transforms.Compose([self.to_tensor,
                                                pair_transforms.RandomResizeCrop((self.height, self.width), factor=(0.75, 1.5)),
                                                pair_transforms.RandomGamma(min=0.8, max=1.2),
                                                pair_transforms.RandomBrightness(min=0.5, max=2.0),
                                                pair_transforms.RandomColorBrightness(min=0.8, max=1.2)])
        else:
            self.data_aug = transforms.Compose([self.to_tensor,
                                                pair_transforms.Resize((self.height, self.width)),
                                                pair_transforms.RandomGamma(min=0.8, max=1.2),
                                                pair_transforms.RandomBrightness(min=0.5, max=2.0),
                                                pair_transforms.RandomColorBrightness(min=0.8, max=1.2)])
            
        self.no_data_aug = transforms.Compose([self.to_tensor,
                                               pair_transforms.Resize((self.height, self.width))])
        
        
        """
        we do colmap in _getitem_ at first, but it will leadding dataloader return none
        so we remove those no colmap data
        """
        if self.use_colmap:
            new_filenames = []
            for index in range(len(self.filenames)):
                line = self.filenames[index].split()
                folder = line[0]
                if len(line) == 3:
                    frame_index = int(line[1])
                else:
                    frame_index = 0
                # c_path = self.get_image_path(folder, frame_index, "l")
                # c_path = c_path.replace(self.data_path, self.colmap_path).replace("/image_02/data/", "/").replace(self.img_ext, "/")
                pose_path = os.path.join(self.colmap_path, folder, "{:010d}".format(frame_index))
                if os.path.exists(os.path.join(pose_path, "poses.npy")) and os.path.exists(os.path.join(pose_path, "poses_flip.npy")):
                    new_filenames.append(self.filenames[index])
            self.filenames = new_filenames

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im)] = self.resize(inputs[(n, im, i)])
                inputs[(n + "_aug", im)] = color_aug(inputs[(n, im)])
                inputs[(n, im)] = self.to_tensor(inputs[(n, im)])
                inputs[(n + "_aug", im)] = self.to_tensor(inputs[(n + "_aug", im)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.
        """
        inputs = {}

        do_data_aug = self.is_train
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if do_flip:
            inputs[("color", "r", -1)] = self.get_color(folder, frame_index, "l", do_flip)
            inputs[("color", "l", -1)] = self.get_color(folder, frame_index, "r", do_flip)
            for novel_frame_id in self.novel_frame_ids:
                inputs[("color", novel_frame_id, -1)] = self.get_color(folder, frame_index+novel_frame_id, "r", do_flip)
        else:
            inputs[("color", "l", -1)] = self.get_color(folder, frame_index, "l", do_flip)
            inputs[("color", "r", -1)] = self.get_color(folder, frame_index, "r", do_flip)
            for novel_frame_id in self.novel_frame_ids:
                inputs[("color", novel_frame_id, -1)] = self.get_color(folder, frame_index+novel_frame_id, "l", do_flip)

            
        self.load_depth = self.check_depth(index)
        if self.load_depth:
            if do_flip:
                inputs[("depth_gt", "r")] = torch.from_numpy(np.expand_dims(self.get_depth(folder, frame_index, "l", do_flip), 0).astype(np.float32))
                inputs[("depth_gt", "l")] = torch.from_numpy(np.expand_dims(self.get_depth(folder, frame_index, "r", do_flip), 0).astype(np.float32))
            else:
                inputs[("depth_gt", "l")] = torch.from_numpy(np.expand_dims(self.get_depth(folder, frame_index, "l", do_flip), 0).astype(np.float32))
                inputs[("depth_gt", "r")] = torch.from_numpy(np.expand_dims(self.get_depth(folder, frame_index, "r", do_flip), 0).astype(np.float32))
            
            
        if do_data_aug:
            inputs = self.data_aug(inputs)
        else:
            inputs = self.no_data_aug(inputs)

        # for i in ["l", "r"] + self.novel_frame_ids:
        #     del inputs[("color", i, -1)]
            # inputs[("color", i, -1)] = self.no_data_aug(inputs[("color", i, -1)])

        K = self.K.copy()

        K[0, :] *= self.width
        K[1, :] *= self.height

        inv_K = np.linalg.pinv(K)

        inputs["K"] = torch.from_numpy(K)
        inputs["inv_K"] = torch.from_numpy(inv_K)

        stereo_T_l = np.eye(4, dtype=np.float32)
        stereo_T_l[0, 3] = 0.1
        stereo_T_r = np.eye(4, dtype=np.float32)
        stereo_T_r[0, 3] = -0.1

        # All (Rt,t) represent the view change from t to left, except "l"
        # "l" represent the view change from left to right
        inputs[("Rt", "l")] = torch.from_numpy(stereo_T_l)
        inputs[("Rt", "r")] = torch.from_numpy(stereo_T_r)
        
        # for k,v in os.environ.items():
        #     print(k, v)
        # quit()
        
        if self.use_colmap:
            try:
                image_paths = {}
                image_paths[(0, "l")] = self.get_image_path(folder, frame_index, "l")
                image_paths[(0, "r")] = self.get_image_path(folder, frame_index, "r")
                for novel_frame_id in self.novel_frame_ids:
                    image_paths[(novel_frame_id, "l")] = self.get_image_path(folder, frame_index+novel_frame_id, "l")
                    image_paths[(novel_frame_id, "r")] = self.get_image_path(folder, frame_index+novel_frame_id, "r")
                    
                colmap_path = image_paths[(0, "l")].replace(self.data_path, self.colmap_path).replace("/image_02/data/", "/").replace(self.img_ext, "/")
                if not os.path.exists(colmap_path):
                    colmap_image_path = os.path.join(colmap_path, "images/")
                    os.makedirs(colmap_image_path)
                    for k,v in image_paths.items():
                        k0, k1 = k
                        shutil.copyfile(v, os.path.join(colmap_image_path, str(k0)+k1+self.img_ext))
                    subprocess.run("colmap feature_extractor --database_path " + os.path.join(colmap_path, "database.db") + " --image_path " + colmap_image_path + " --ImageReader.camera_model PINHOLE " + \
                            "--ImageReader.camera_params 720.36,720,621,187.5 --SiftExtraction.max_image_size 4096 --ImageReader.single_camera 1", stdout=subprocess.DEVNULL, shell=True, check=True)
                    subprocess.run("colmap exhaustive_matcher --database_path " + os.path.join(colmap_path, "database.db") + "  --SiftMatching.confidence 0.85 --SiftMatching.min_num_inliers 5", stdout=subprocess.DEVNULL, shell=True, check=True)# --Mapper.max_model_overlap 40 --Mapper.min_model_size 6
                    os.makedirs(os.path.join(colmap_path, "sparse/"))
                    subprocess.run("colmap mapper --database_path " + os.path.join(colmap_path, "database.db") + " --image_path " + colmap_image_path + " --output_path " + colmap_path + " --Mapper.init_max_forward_motion 1 --Mapper.init_min_tri_angle 0.25", stdout=subprocess.DEVNULL, shell=True, check=True)
                    subprocess.run("colmap model_converter --input_path " + os.path.join(colmap_path, "0/") + " --output_path " + colmap_path + " --output_type TXT", stdout=subprocess.DEVNULL, shell=True, check=True)
                    shutil.rmtree(colmap_image_path)
                    poses_original, poses_flip = self.rectify_poses(os.path.join(colmap_path, "images.txt"))
                    #save
                    # poses_original_json = json.dumps(poses_original)
                    # with open(os.path.join(colmap_path, "poses.json"), "w") as f:  
                    #     f.write(poses_original_json)
                    #     f.close()
                    # poses_flip_json = json.dumps(poses_flip)
                    # with open(os.path.join(colmap_path, "poses_flip.json"), "w") as f:  
                    #     f.write(poses_flip_json)
                    #     f.close()
                    np.save(os.path.join(colmap_path, "poses.npy"), poses_original)
                    np.save(os.path.join(colmap_path, "poses_flip.npy"), poses_flip)
                    
                if do_flip:
                    # pose_file = open(os.path.join(colmap_path, "poses_flip.json"), "r")
                    # poses = json.load(pose_file)
                    poses = np.load(os.path.join(colmap_path, "poses.npy"), allow_pickle=True).item()
                    inputs.update(poses)
                else:
                    # pose_file = open(os.path.join(colmap_path, "poses.json"), "r")
                    # poses = json.load(pose_file)
                    poses = np.load(os.path.join(colmap_path, "poses_flip.npy"), allow_pickle=True).item()
                    inputs.update(poses)
            except:
                return None
                for k,v in inputs.items():
                    inputs[k] = None
                for frame_id in self.novel_frame_ids:
                    inputs[("Rt", frame_id)] = None

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    
    def get_image_path(self, folder, frame_index, side):
        raise NotImplementedError
    
    def rectify_poses(self, path):
        poses = {}
        lines = readlines(path)
        for line in lines:
            line = line.split()
            if len(line) == 10 and line[-1][-4:] == self.img_ext:
                R = self.qvec2rotmat(list(map(float, line[1:5])))
                t = np.array(list(map(float, line[5:8])), dtype=np.float32)
                frame_id = int(line[-1][:-5])
                side = line[-1][-5]
                Rt = np.eye(4)
                Rt[:3, :3] = R
                Rt[:3, 3] = t
                poses[(frame_id, side)] = Rt
                
        # No flip, Rt=Rt*Rt^{-1}_l
        Rts_inv = np.linalg.inv(poses[(0, "l")])
        Rt_r = np.matmul(poses[(0, "r")], Rts_inv)
        t_r = Rt_r[:3, 3]
        scale_f = np.linalg.norm(t_r, ord=2) / 0.1
        poses_original = {}
        for frame_id in self.novel_frame_ids:
            poses_original[("Rt", frame_id)] = np.matmul(poses[(frame_id, "l")], Rts_inv)
            poses_original[("Rt", frame_id)][:3, 3] /= scale_f
            
        #flip
        Rts_inv = np.linalg.inv(poses[(0, "r")])
        Rt_l = np.matmul(poses[(0, "l")], Rts_inv)
        t_l = Rt_l[:3, 3]
        scale_f = np.linalg.norm(t_l, ord=2) / 0.1
        poses_flip = {}
        for frame_id in self.novel_frame_ids:
            poses_flip[("Rt", frame_id)] = np.matmul(poses[(frame_id, "r")], Rts_inv)
            poses_flip[("Rt", frame_id)][:3, 3] /= scale_f
            poses_flip[("Rt", frame_id)][0, 1:] *= -1.
            poses_flip[("Rt", frame_id)][1:, 0] *= -1.
        return poses_original, poses_flip
    
    def qvec2rotmat(self, qvec):
        return np.array([
            [1 - 2 * float(qvec[2])**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]], dtype=np.float32)
