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

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json
import random

@dataclass
class TranspNerfDataParserBlenderConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: TranspNerfData)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""


@dataclass
class TranspNerfData(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: TranspNerfDataParserConfig

    def __init__(self, config: TranspNerfDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        if self.alpha_color is not None:
            self.alpha_color_tensor = get_color(self.alpha_color)
        else:
            self.alpha_color_tensor = None

    def _generate_dataparser_outputs(self, split="test"): # test set which includes normals and depths

        print("Generating blender dataset.")

        meta = load_from_json(self.data)
        data_dir = self.data.parent
        image_filenames = []
        normal_filenames = []
        depth_filenames = []
        depth_scale = 1
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        # render = "hotdog"
        # render = "ficus"
        render = "wine"
        #render = "coffee"

        if render == "hotdog":
            data_id = "0029" #hotdog - 200 captures
        elif render == "ficus":
            data_id = "0136" # ficus - 200 captures
        elif render == "wine":
            data_id = "0090" #transpwine - 40 captures
            scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1], [1.5, 1.5, 2]], dtype=torch.float32)) #shift box z axis
        elif render == "coffee":
            data_id = "0000"  #coffee - 40 captures
            depth_scale = 0.1
        else:
            print("error")
        

        poses = []
        for frame in random.sample(meta["frames"], 40): #meta["frames"]:
            fname = data_dir / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)

            # added normal and depths
            fname_depth = data_dir / Path(frame["file_path"].replace("./", "") + "_depth_" + data_id + ".png")
            fname_normal = data_dir / Path(frame["file_path"].replace("./", "") + "_normal_" + data_id + ".png")
            normal_filenames.append(fname_normal)
            depth_filenames.append(fname_depth)
            
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "normal_filenames": normal_filenames if len(normal_filenames) > 0 else None,
                "depth_unit_scale_factor": depth_scale,
                #"mask_color": self.config.mask_color,
            },
        )

        return dataparser_outputs