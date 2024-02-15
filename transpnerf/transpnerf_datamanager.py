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
        ray_bundle = bundle.to(self.device)
        return ray_bundle, batch


#here --- batch containes data - can append to ray bundle!!!!! AHHHHHHHH
