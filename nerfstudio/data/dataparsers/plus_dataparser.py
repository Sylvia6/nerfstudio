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
import pickle

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox


opencv2opengl = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)
        

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')
        
@dataclass
class PlusDataParserConfig(DataParserConfig):
    """NuScenes dataset config.
    NuScenes (https://www.nuscenes.org/nuscenes) is an autonomous driving dataset containing 1000 20s clips.
    Each clip was recorded with a suite of sensors including 6 surround cameras.
    It also includes 3D cuboid annotations around objects.
    We optionally use these cuboids to mask dynamic objects by specifying the mask_dir flag.
    To create these masks use nerfstudio/scripts/datasets/process_nuscenes_masks.py.
    """

    _target: Type = field(default_factory=lambda: Plus)
    """target class to instantiate"""
    data: Path = Path("/mnt/intel/artifact_management/auto-labeling-multi-frame/j7_10-multi-frame_manual_all_bag/000210_20211111T153653_j7-00010_42_1to21.db")  # TODO: rename to scene but keep checkpoint saving name?
    """Name of the scene."""
    # seq: str = '016'
    cameras: Tuple[Literal["back", "front", "front_left", "front_right", "left", "right"], ...] = ("front_left")
    """Which cameras to use."""
    train_split_fraction: float = 1. 
    # 0.9
    """The percent of images to use for training. The remaining images are for eval."""

@dataclass
class Plus(DataParser):
    """Plus DatasetParser"""

    config: PlusDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        image_filenames = sorted(glob(f'{self.config.data}/{self.config.cameras}/*.png'))
        
        calibration_file = os.path.join(self.config.data, "calib", ("%06d"%0)+".pkl")
        calib = load_pickle(calibration_file)
        K = calib['P1']
        cam_i_imu =  calib[f"Tr_cam_to_imu_{self.config.cameras}"]
        
        poses = []
        for frame in range(0, len(image_filenames)):
            info = load_pickle(os.path.join(self.config.data, 'ego_pos_with_vel', ("%06d"%frame)+".pkl"))
            poses.append(info['ego_pose'])
        c2w = np.array(poses) @ cam_i_imu @ opencv2opengl
        
        timestamp_path = os.path.join(self.config.data, "timestamp")
        timestamps = np.array([load_pickle(os.path.join(timestamp_path, ("%06d"%frame)+".pkl")) for frame in range(0, len(image_filenames))])
        times = torch.tensor(timestamps - timestamps[0], dtype=torch.float32)
                
        image_size = cv2.imread(image_filenames[0]).shape[:2]
        
        c2w = torch.from_numpy(c2w.astype(np.float32))
        # center poses
        c2w[:, :3, 3] -= c2w[:, :3, 3].mean(dim=0)
        # scale poses
        c2w[:, :3, 3] /= c2w[:, :3, 3].abs().max()
        
        cameras = Cameras(
            camera_to_worlds=c2w[:, :3, :],
            fx=K[0,0],
            fy=K[1,1],
            cx=K[0,2],
            cy=K[1,2],
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
        