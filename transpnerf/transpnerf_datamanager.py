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

@dataclass
class TranspNerfDataManagerConfig(VanillaDataManagerConfig):
    """TranspNerf DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: TranspNerfDataManager)


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
    
    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        # print("self.dataset_type -->", self.dataset_type)  # for now, hardcoding since the datset type is not changing

        return DepthNormalDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )
    
    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return DepthNormalDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )
  
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        print("image_batch image shape: ", image_batch["image"].shape)
        print("image_batch image_idx shape: ", image_batch["image_idx"].shape)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        
        ray_indices = batch["indices"]
        print(" ray_indicies shape" ,  ray_indices.shape)
        ray_bundle = self.train_ray_generator(ray_indices)
        print("budle metadata keys before: ", ray_bundle.metadata.keys())
        ray_bundle.metadata["depth"] = image_batch["depth_image"]
        print("budle metadata keys after: ", ray_bundle.metadata.keys())
        print("shape directions_norm: ", ray_bundle.metadata["directions_norm"].shape)
        print("shape depth: ", ray_bundle.metadata["depth"].shape)
        return ray_bundle, batch

