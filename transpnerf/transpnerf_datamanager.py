"""
TranspNerf DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Generic

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.parallel_datamanager import (
    ParallelDataManager,
    ParallelDataManagerConfig,
)
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.cameras.cameras import Cameras
from typing_extensions import TypeVar
from nerfstudio.data.datasets.base_dataset import InputDataset
from transpnerf.depth_normal_dataset import DepthNormalDataset
import random

@dataclass
class TranspNerfDataManagerConfig(VanillaDataManagerConfig):
    """TranspNerf DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: TranspNerfDataManager)
    use_zoe_depth: bool = False
    """Weather to use Zoe depth or not"""

TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)

class TranspNerfDataManager(VanillaDataManager, Generic[TDataset]):
    """TranspNerf DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: TranspNerfDataManagerConfig

    def __init__(
        self,
        config: TranspNerfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )


    def create_train_dataset(self):
        """Sets up the data loaders for training"""
        # for now, hardcoding since the datset type is not changing

        return DepthNormalDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            use_zoe_depth=self.config.use_zoe_depth,
        )
    
    def create_eval_dataset(self):
        """Sets up the data loaders for evaluation"""
        return DepthNormalDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            use_zoe_depth=self.config.use_zoe_depth,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)

        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        # add on normal and depth metadata
        if "depth_image" in image_batch:
            ray_bundle.metadata["depth"] =  self._process_depth_normal_metadata(ray_indices, image_batch["depth_image"])
        if "normal_image" in image_batch:
            ray_bundle.metadata["normal"] = self._process_depth_normal_metadata(ray_indices, image_batch["normal_image"])
        
        return ray_bundle, batch

    def _process_depth_normal_metadata(self, ray_indices: torch.Tensor, data: torch.Tensor) -> torch.tensor:
        # this code is exactly what is happening in: 
        # https://github.com/nerfstudio-project/nerfstudio/blob/45db2bcfabe6e0644a3a45a50ed80a9a685ddc34/nerfstudio/data/pixel_samplers.py#L239

        c, y, x = (i.flatten() for i in torch.split(ray_indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        return data[c, y, x] # x, y

