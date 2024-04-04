"""
TranspNerf Model File
"""
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from typing import Dict, Tuple, Type, Optional
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.fields.nerfacto_field import NerfactoField
import math
from transpnerf.transpnerf_field import TranspNerfField
from nerfstudio.model_components.losses import DepthLossType, depth_loss, depth_ranking_loss
from nerfstudio.utils import colormaps
from nerfstudio.model_components import losses
import numpy as np
from nerfstudio.field_components.spatial_distortions import SceneContraction

@dataclass
class TranspNerfModelConfig(NerfactoModelConfig):
    """TranspNerf Model Configuration.
    """

    _target: Type = field(default_factory=lambda: TranspNerfModel)
    
    # feild related
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""
    use_appearance_embedding: bool = True
    """Whether to use an appearance embedding."""
    average_init_density: float = 1.0
    """Average initial density output from MLP. """
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    
    # depth-nerfacto related
    depth_loss_mult: float = 1e-3
    """Lambda of the depth loss."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.01
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    should_decay_sigma: bool = False
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""
    depth_loss_type: DepthLossType = DepthLossType.DS_NERF
    """Depth loss type."""

    # transparent related
    apply_refl: bool = True
    """ Weather to apply reflection equations."""
    calc_fresnel: bool = False
    """ Weather to also apply the Fresnel constant"""
    apply_depth_supervision: bool = False
    """ Weather to apply depth-nerfacto depth supervision"""
    

class TranspNerfModel(NerfactoModel):
    """TranspNerf Model."""

    config: TranspNerfModelConfig

    def populate_modules(self):
        super().populate_modules()

        print("----- REFLECT Parameters --- reflection: ",  self.config.apply_refl, " frensel: ", self.config.calc_fresnel, " depth supervision: ", self.config.apply_depth_supervision)

        if self.config.apply_refl and self.config.calc_fresnel:
            if self.config.disable_scene_contraction:
                scene_contraction = None
            else:
                scene_contraction = SceneContraction(order=float("inf"))

        # for fresnel field instantiation
        if self.config.apply_refl and self.config.calc_fresnel:  
            appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

            self.field = TranspNerfField(
                aabb=self.scene_box.aabb,
                hidden_dim=self.config.hidden_dim,
                num_levels=self.config.num_levels,
                max_res=self.config.max_res,
                base_res=self.config.base_res,
                features_per_level=self.config.features_per_level,
                log2_hashmap_size=self.config.log2_hashmap_size,
                hidden_dim_color=self.config.hidden_dim_color,
                hidden_dim_transient=self.config.hidden_dim_transient,
                spatial_distortion=scene_contraction,
                num_images=self.num_train_data,
                use_pred_normals=self.config.predict_normals,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
                appearance_embedding_dim=appearance_embedding_dim,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
        
        # for depth-nerfacto 
        if self.config.apply_depth_supervision:
            if self.config.should_decay_sigma:
                self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
            else:
                self.depth_sigma = torch.tensor([self.config.depth_sigma])

    def _adjust_normal(self, normals, in_dir):
        ## taken from https://github.com/dawning77/NeRRF
        in_dot = (in_dir * normals).sum(dim=-1)
        mask = in_dot > 0
        normals[mask] = -normals[mask]  # make sure normal point to in_dir
        return normals

    def _fresnel(self, n, in_dir, out_dir, normal):
        ## taken from https://github.com/dawning77/NeRRF
        in_dot = (in_dir * normal).sum(-1)
        out_dot = (out_dir * normal).sum(-1)

        F = ((in_dot - n * out_dot) / (in_dot + n * out_dot)) ** 2 + (
            (n * in_dot - out_dot) / (n * in_dot + out_dot)
        ) ** 2
        return F / 2
    
    def _reflection(self, ray_bundle: RayBundle, calc_fresnel: bool):
        input_origins = ray_bundle.origins
        input_directions = ray_bundle.directions
        depth = ray_bundle.metadata["depth"]
        normal = ray_bundle.metadata["normal"]
        if calc_fresnel:
            normal = self._adjust_normal(ray_bundle.metadata["normal"], input_directions)
       
        # Generate a mask to exclude the background
        target_tensor = torch.tensor([0., 0., 0.]).cuda()  # target black normal (background)
        index_mask = torch.all(normal != target_tensor, dim=1) # indicies that are not background

        #index_mask = torch.all(depth != 0, dim=1)

        #print("index mask True count --> ", torch.sum(index_mask.int()).item())
        # calculate incident angle
        cos_theta_i = (-input_directions * normal).sum(dim=1) 

        # calculate reflected rays origins and directions
        refl_origins = input_origins + depth.expand(-1, 3) * input_directions
        refl_dir = input_directions + 2 * cos_theta_i.unsqueeze(-1) * normal
        refl_dir = refl_dir / torch.norm(refl_dir, dim=-1).unsqueeze(-1)

        # test move origin
        #refl_origins = refl_origins - depth.expand(-1, 3) * refl_dir

        # use only reflected part for now
        ray_bundle.origins[index_mask] = refl_origins.clone()[index_mask]
        ray_bundle.directions[index_mask] = refl_dir.clone()[index_mask].to(ray_bundle.directions.dtype)

        #fresenel 
        if calc_fresnel:
            ior = 1.5 #1
            ior_ = 1/ior
            cos_theta_o = torch.sqrt(1 - (ior_**2) * (1- cos_theta_i**2))
            refract_dir_1 = ior_ * input_directions + (ior_*cos_theta_i - cos_theta_o).unsqueeze(-1)*normal
            refract_dir_1 = refract_dir_1 / torch.norm(refract_dir_1, dim=-1).unsqueeze(-1)
            fresnel_1 = self._fresnel(ior, input_directions, refract_dir_1, normal)
            return ray_bundle, fresnel_1, index_mask

        return ray_bundle, index_mask

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)

        fresnel_info = {}
        index_mask = None
        fresnel_1 = None
        
        if self.config.apply_refl:
            # apply reflection
            if "depth" in ray_bundle.metadata.keys() and "normal" in ray_bundle.metadata.keys():
                if self.config.calc_fresnel:
                    ray_bundle, fresnel_1, index_mask = self._reflection(ray_bundle, self.config.calc_fresnel)
                    fresnel_info = {"fresnel": fresnel_1, "index_mask": index_mask}
                else:
                    ray_bundle, index_mask = self._reflection(ray_bundle, self.config.calc_fresnel)
        
        # proposal sampler
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        
        if self.config.apply_refl and self.config.calc_fresnel:
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals, fresnel_info=fresnel_info)
        else:
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        
        if self.config.apply_depth_supervision and ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs
    
    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)

        if self.config.apply_depth_supervision and self.training:
            if (
                losses.FORCE_PSEUDODEPTH_LOSS
                and self.config.depth_loss_type not in losses.PSEUDODEPTH_COMPATIBLE_LOSSES
            ):
                raise ValueError(
                    f"Forcing pseudodepth loss, but depth loss type ({self.config.depth_loss_type}) must be one of {losses.PSEUDODEPTH_COMPATIBLE_LOSSES}"
                )
            if self.config.depth_loss_type in (DepthLossType.DS_NERF, DepthLossType.URF):
                metrics_dict["depth_loss"] = 0.0
                sigma = self._get_sigma().to(self.device)
                termination_depth = batch["depth_image"].to(self.device)
                for i in range(len(outputs["weights_list"])):
                    metrics_dict["depth_loss"] += depth_loss(
                        weights=outputs["weights_list"][i],
                        ray_samples=outputs["ray_samples_list"][i],
                        termination_depth=termination_depth,
                        predicted_depth=outputs["expected_depth"],
                        sigma=sigma,
                        directions_norm=outputs["directions_norm"],
                        is_euclidean=self.config.is_euclidean_depth,
                        depth_loss_type=self.config.depth_loss_type,
                    ) / len(outputs["weights_list"])
            elif self.config.depth_loss_type in (DepthLossType.SPARSENERF_RANKING,):
                metrics_dict["depth_ranking"] = depth_ranking_loss(
                    outputs["expected_depth"], batch["depth_image"].to(self.device)
                )
            else:
                raise NotImplementedError(f"Unknown depth loss type {self.config.depth_loss_type}")

        return metrics_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.config.apply_depth_supervision and self.training:
            assert metrics_dict is not None and ("depth_loss" in metrics_dict or "depth_ranking" in metrics_dict)
            if "depth_ranking" in metrics_dict:
                loss_dict["depth_ranking"] = (
                    self.config.depth_loss_mult
                    * np.interp(self.step, [0, 2000], [0, 0.2])
                    * metrics_dict["depth_ranking"]
                )
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
        return loss_dict
    
    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics, images = super().get_image_metrics_and_images(outputs, batch)

        if self.config.apply_depth_supervision:
            ground_truth_depth = batch["depth_image"].to(self.device)
            if not self.config.is_euclidean_depth:
                ground_truth_depth = ground_truth_depth * outputs["directions_norm"]

            ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
            predicted_depth_colormap = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
                near_plane=float(torch.min(ground_truth_depth).cpu()),
                far_plane=float(torch.max(ground_truth_depth).cpu()),
            )
            images["depth"] = torch.cat([ground_truth_depth_colormap, predicted_depth_colormap], dim=1)
            depth_mask = ground_truth_depth > 0
            metrics["depth_mse"] = float(
                torch.nn.functional.mse_loss(outputs["depth"][depth_mask], ground_truth_depth[depth_mask]).cpu()
            )
        return metrics, images
    
    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma
