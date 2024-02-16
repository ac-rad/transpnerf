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

"""
Depth Normal dataset.
"""

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image
from rich.progress import track

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.model_components import losses
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
import torch.nn.functional as F
from torch.functional import norm

class DepthNormalDataset(InputDataset):
    """Dataset that returns images and depths. If no depths are found, then we generate them with Zoe Depth.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        print("setting up depth normal dataset!")

        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]

    def get_metadata(self, data: Dict) -> Dict:
        if self.depth_filenames is None:
            print("no depth filenames")
            return {}#return {"depth_image": self.depths[data["image_idx"]]} - for zoe depth
        
        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        )

        normal_image = self._compute_normals(depth_image)

        return {"depth_image": depth_image, "normal_image": normal_image}

    def _compute_normals(self, depths: torch.Tensor) -> torch.Tensor:
        # this code is from https://github.com/Ruthrash/surface_normal_filter 

        # original depth shape is (800, 800, 1)
        depths_reshape = depths.permute(2, 0, 1).unsqueeze(1) # (1, 1, 800, 800)
        nb_channels = 1

        delzdelxkernel = torch.tensor([[0.00000, 0.00000, 0.00000],
                                        [-1.00000, 0.00000, 1.00000],
                                        [0.00000, 0.00000, 0.00000]], dtype=torch.float64)
        delzdelxkernel = delzdelxkernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
        delzdelx = F.conv2d(depths_reshape, delzdelxkernel)

        delzdelykernel = torch.tensor([[0.00000, -1.00000, 0.00000],
                                        [0.00000, 0.00000, 0.00000],
                                        [0.0000, 1.00000, 0.00000]], dtype=torch.float64)
        delzdelykernel = delzdelykernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)

        delzdely = F.conv2d(depths_reshape, delzdelykernel)

        delzdelz = torch.ones(delzdely.shape, dtype=torch.float64)

        surface_norm = torch.stack((-delzdelx,-delzdely, delzdelz),2)
        print("surface_norm.shape --> ", surface_norm.shape)
        surface_norm = torch.div(surface_norm,  norm(surface_norm, dim=2)[:,:,None,:,:])

        print("surface_norm.shape 2 --> ", surface_norm.shape)
        surface_norm = surface_norm.squeeze().permute(1, 2, 0) #(800, 800, 3)
        print("surface_norm.shape after--> ", surface_norm.shape)

        return surface_norm
