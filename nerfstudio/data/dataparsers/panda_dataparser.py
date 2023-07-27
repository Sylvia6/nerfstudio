# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for NuScenes dataset"""
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Type

import numpy as np
import pyquaternion
import torch

from glob import glob
import json, cv2
from pandaset import DataSet as PandaSet

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox


def rotation_translation_to_pose(p):
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""
    r_quat = [v for v in p['heading'].values()]
    t_vec = [v for v in p['position'].values()]
    pose = np.eye(4)

    # NB: Nuscenes recommends pyquaternion, which uses scalar-first format (w x y z)
    # https://github.com/nutonomy/nuscenes-devkit/issues/545#issuecomment-766509242
    # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L299
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    pose[:3, :3] = pyquaternion.Quaternion(r_quat).rotation_matrix
    pose[:3, 3] = t_vec
    return pose

def nerf_matrix_to_ngp(pose, scale=1):       
    new_poses = np.array(
        [
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return new_poses


@dataclass
class PandaDataParserConfig(DataParserConfig):
    """NuScenes dataset config.
    NuScenes (https://www.nuscenes.org/nuscenes) is an autonomous driving dataset containing 1000 20s clips.
    Each clip was recorded with a suite of sensors including 6 surround cameras.
    It also includes 3D cuboid annotations around objects.
    We optionally use these cuboids to mask dynamic objects by specifying the mask_dir flag.
    To create these masks use nerfstudio/scripts/datasets/process_nuscenes_masks.py.
    """

    _target: Type = field(default_factory=lambda: Pandar)
    """target class to instantiate"""
    data: Path = Path("/mnt/intel/jupyterhub/lilu/pandaset")  # TODO: rename to scene but keep checkpoint saving name?
    """Name of the scene."""
    seq: str = '016'
    cameras: Tuple[Literal["back", "front", "front_left", "front_right", "left", "right"], ...] = ("front")
    """Which cameras to use."""
    train_split_fraction: float = 1. 
    # 0.9
    """The percent of images to use for training. The remaining images are for eval."""

@dataclass
class Pandar(DataParser):
    """Pandar DatasetParser"""

    config: PandaDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        panda = PandaSet(self.config.data)[self.config.seq]
        camera_path = f'{self.config.data}/{self.config.seq}/camera/{self.config.cameras}_camera'
        image_filenames = sorted(glob(f'{camera_path}/*.jpg')) 
        with open(f'{camera_path}/poses.json', 'r') as f:
            pose_data = json.load(f)
            c2w = np.array([
                nerf_matrix_to_ngp(rotation_translation_to_pose(p)) for p in pose_data
            ])
        with open(f'{camera_path}/intrinsics.json', 'r') as f:
            cam_data = json.load(f)
            K = np.array(list(cam_data.values()))            
        with open(f'{self.config.data}/{self.config.seq}/meta/timestamps.json', 'r') as f:    
            time_data = np.array(json.load(f))
        
        times = torch.tensor(time_data - time_data[0], dtype=torch.float32)
                
        image_size = cv2.imread(image_filenames[0]).shape[:2]
        
        c2w = torch.from_numpy(c2w)
        # center poses
        c2w[:, :3, 3] -= c2w[:, :3, 3].mean(dim=0)
        # scale poses
        c2w[:, :3, 3] /= c2w[:, :3, 3].abs().max()
        
        cameras = Cameras(
            camera_to_worlds=c2w[:, :3, :],
            fx=K[0],
            fy=K[1],
            cx=K[2],
            cy=K[3],
            height=image_size[0],
            width=image_size[1],
            camera_type=CameraType.PERSPECTIVE,            
            times = times
        )
        aabb_scale = 1.0
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
        dataparser_outputs = DataparserOutputs(
            image_filenames = image_filenames,
            cameras = cameras,
            scene_box = scene_box
        )
        return dataparser_outputs
        