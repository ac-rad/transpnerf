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
# from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.model_components import losses
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
import torch.nn.functional as F
from torch.functional import norm
import cv2

def get_depth_normal_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
    isDepth: bool = True,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width, 1].
    """
    if isDepth:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float32) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
        return torch.from_numpy(image[:, :, np.newaxis])
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR) 
        image = image.astype(np.float32) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
        return torch.squeeze(torch.from_numpy(image[:, :, np.newaxis]))


class DepthNormalDataset(InputDataset):
    """Dataset that returns images and depths. If no depths are found, then we generate them with Zoe Depth.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, use_zoe_depth: bool = False):
        super().__init__(dataparser_outputs, scale_factor)

        print("setting up depth normal dataset!")

        self.depth_filenames = []
        self.normal_filenames = []

        if "depth_filenames" in self.metadata:
            self.depth_filenames = self.metadata["depth_filenames"]
        if "normal_filenames" in self.metadata:
            self.normal_filenames = self.metadata["normal_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]

        self.use_zoe_depth = use_zoe_depth
        if self.use_zoe_depth:
            self._compute_zoe_depth(dataparser_outputs)

    def get_metadata(self, data: Dict) -> Dict:

        #if zoe depth
        if self.use_zoe_depth:
            depth_image = self.depths[data["image_idx"]]
            normal_image = self._compute_normals(depth_image)
        
        # use file depths
        else:
            if self.depth_filenames is None:
                print("ERROR !!!! -- no depth files")
                return {}
            
            height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
            width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])
            scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale

            # Scale depth images to meter units and also by scaling applied to cameras
            filepath_depth = self.depth_filenames[data["image_idx"]]
            depth_image = get_depth_normal_image_from_path(
                filepath=filepath_depth, height=height, width=width, scale_factor=scale_factor, isDepth=True
            )
            
            # load normal
            if self.normal_filenames:
                filepath_normal = self.normal_filenames[data["image_idx"]]
                normal_image = get_depth_normal_image_from_path(
                    filepath=filepath_normal, height=height, width=width, scale_factor=scale_factor, isDepth=False
                )
            # compute from depth
            else:
                normal_image = self._compute_normals(depth_image)
            
        return {"depth_image": depth_image, "normal_image": normal_image} 

    def _compute_normals(self, depths: torch.Tensor) -> torch.Tensor:
        # this code is from https://github.com/Ruthrash/surface_normal_filter with a few modifications

        depths_reshape = depths.permute(2, 0, 1).unsqueeze(1) 
        nb_channels = 1

        delzdelxkernel = torch.tensor([[0.00000, 0.00000, 0.00000],
                                        [-1.00000, 0.00000, 1.00000],
                                        [0.00000, 0.00000, 0.00000]]) 
        delzdelxkernel = delzdelxkernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
        delzdelx = F.conv2d(depths_reshape, delzdelxkernel, padding=1)

        delzdelykernel = torch.tensor([[0.00000, -1.00000, 0.00000],
                                        [0.00000, 0.00000, 0.00000],
                                        [0.0000, 1.00000, 0.00000]]) 
        delzdelykernel = delzdelykernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)

        delzdely = F.conv2d(depths_reshape, delzdelykernel, padding=1)

        delzdelz = torch.ones(delzdely.shape) 

        surface_norm = torch.stack((-delzdelx,-delzdely, delzdelz),2)
        surface_norm = torch.div(surface_norm,  norm(surface_norm, dim=2)[:,:,None,:,:])

        surface_norm = surface_norm.squeeze().permute(1, 2, 0) 

        return surface_norm
    
    def _compute_zoe_depth(self, dataparser_outputs):
        # taken from nerfacto's depth dataset: https://github.com/nerfstudio-project/nerfstudio/blob/05ce76db9902827d2d9ad1556f8d3f563c7daa11/nerfstudio/data/datasets/depth_dataset.py#L48
        
        device ="cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        CONSOLE.print("[bold yellow] No depth data found! Generating pseudodepth...")
        losses.FORCE_PSEUDODEPTH_LOSS = True
        CONSOLE.print("[bold red] Using psueodepth: forcing depth loss to be ranking loss.")
        cache = dataparser_outputs.image_filenames[0].parent / "depths.npy"
        # Note: this should probably be saved to disk as images, and then loaded with the dataparser.
        #  That will allow multi-gpu training.
        if cache.exists():
            CONSOLE.print("[bold yellow] Loading pseudodata depth from cache!")
            # load all the depths
            self.depths = np.load(cache)
            self.depths = torch.from_numpy(self.depths).to(device)
        else:
            depth_tensors = []
            transforms = self._find_transform(dataparser_outputs.image_filenames[0])
            data = dataparser_outputs.image_filenames[0].parent
            if transforms is not None:
                meta = json.load(open(transforms, "r"))
                frames = meta["frames"]
                filenames = [data / frames[j]["file_path"].split("/")[-1] for j in range(len(frames))]
            else:
                meta = None
                frames = None
                filenames = dataparser_outputs.image_filenames

            repo = "isl-org/ZoeDepth"
            self.zoe = torch_compile(torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(device))

            for i in track(range(len(filenames)), description="Generating depth images"):
                image_filename = filenames[i]
                pil_image = Image.open(image_filename)
                image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
                if len(image.shape) == 2:
                    image = image[:, :, None].repeat(3, axis=2)
                image = torch.from_numpy(image.astype("float32") / 255.0)

                with torch.no_grad():
                    image = torch.permute(image, (2, 0, 1)).unsqueeze(0).to(device)
                    if image.shape[1] == 4:
                        image = image[:, :3, :, :]
                    depth_tensor = self.zoe.infer(image).squeeze().unsqueeze(-1)

                depth_tensors.append(depth_tensor)

            self.depths = torch.stack(depth_tensors)
            np.save(cache, self.depths.cpu().numpy())
    
    def _find_transform(self, image_path: Path) -> Union[Path, None]:
        while image_path.parent != image_path:
            transform_path = image_path.parent / "transforms.json"
            if transform_path.exists():
                return transform_path
            image_path = image_path.parent
        return None