"""
TranspNerf DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.parallel_datamanager import (
    ParallelDataManager,
    ParallelDataManagerConfig,
)
from nerfstudio.cameras.cameras import Cameras


@dataclass
class TranspNerfDataManagerConfig(ParallelDataManagerConfig):
    """TranspNerf DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: TranspNerfDataManager)


class TranspNerfDataManager(ParallelDataManager):
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

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the parallel training processes."""
        self.train_count += 1
        bundle, batch = self.data_queue.get()
        print("budle metadata keys before: ", bundle.metadata.keys())
        bundle.metadata["depth"] = batch["depth_image"]
        ray_bundle = bundle.to(self.device)
        print("budle metadata keys after: ", bundle.metadata.keys())
        return ray_bundle, batch
    
    # def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
    #     """Returns the next batch of data from the eval dataloader."""
    #     self.eval_count += 1
    #     image_batch = next(self.iter_eval_image_dataloader)
    #     assert self.eval_pixel_sampler is not None
    #     assert isinstance(image_batch, dict)
    #     batch = self.eval_pixel_sampler.sample(image_batch)
    #     ray_indices = batch["indices"]
    #     ray_bundle = self.eval_ray_generator(ray_indices)
    #     ray_bundle.metadata["depth"] = batch["depth_image"]
    #     return ray_bundle, batch
    
    # def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
    #     """Retrieve the next eval image."""
    #     for camera, batch in self.eval_dataloader:
    #         assert camera.shape[0] == 1

    #         print("adding metadata to camera here ...")
    #         camera.metadata["depth"] = batch["depth_image"]
    #         return camera, batch
    #     raise ValueError("No more eval images")

