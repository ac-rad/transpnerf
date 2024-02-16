"""
Transpnerf Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from transpnerf.transpnerf_datamanager import (
    TranspNerfDataManager,
    TranspNerfDataManagerConfig,
)
from transpnerf.transpnerf_model import TranspNerfModelConfig
# from transpnerf.transpnerf_pipeline import (
#     TranspNerfPipelineConfig,
# )
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from transpnerf.transpnerf_dataparser import TranspNerfDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from transpnerf.depth_normal_dataset import DepthNormalDataset

transpnerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="transpnerf",  # TODO: rename to your own model
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=TranspNerfDataManagerConfig( #ParallelDataManager
                #_target=TranspNerfDataManager[DepthNormalDataset],
                dataparser=TranspNerfDataParserConfig(), #NerfstudioDataParserConfig(), 
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=TranspNerfModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"), #SO3xR3 for nerfstudiodataparser
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Config for TranspNerf",
)
