"""
TranspNerf Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
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

@dataclass
class TranspNerfModelConfig(NerfactoModelConfig):
    """TranspNerf Model Configuration.
    """

    _target: Type = field(default_factory=lambda: TranspNerfModel)
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

    # transparent related
    apply_refl: bool = True
    calc_fresnel: bool = False
    fresnel_version: int = 1 #in_model - 1, in_field - 0
    adjust_normal: bool = False
    

class TranspNerfModel(NerfactoModel):
    """TranspNerf Model."""

    config: TranspNerfModelConfig

    def populate_modules(self):
        super().populate_modules()

        print("----- REFLECT Parameters ---",  self.config.apply_refl, self.config.calc_fresnel, self.config.fresnel_version, self.config.adjust_normal)

        if self.config.apply_refl and self.config.calc_fresnel and self.config.fresnel_version == 0:
            if self.config.disable_scene_contraction:
                scene_contraction = None
            else:
                scene_contraction = SceneContraction(order=float("inf"))
            
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
    
    def _reflection(self, ray_bundle: RayBundle, calc_fresnel: bool, adjust_normal: bool):
        #print("refl")
        input_origins = ray_bundle.origins
        input_directions = ray_bundle.directions
        depth = ray_bundle.metadata["depth"]
        normal = ray_bundle.metadata["normal"]
        if adjust_normal:
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
            ior = 1.5
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
                    ray_bundle, fresnel_1, index_mask = self._reflection(ray_bundle, self.config.calc_fresnel, self.config.adjust_normal)
                    if self.config.fresnel_version == 0:
                        fresnel_info = {"fresnel": fresnel_1, "index_mask": index_mask}
                else:
                    ray_bundle, index_mask = self._reflection(ray_bundle, self.config.calc_fresnel, self.config.adjust_normal)
        
        # proposal sampler
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        
        if self.config.apply_refl and self.config.calc_fresnel and self.config.fresnel_version == 0:
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals, fresnel_info=fresnel_info)
        else:
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        if self.config.apply_refl and self.config.calc_fresnel and self.config.fresnel_version == 1 and index_mask != None and fresnel_1!= None:
            fresnel_final = (1 - fresnel_1.unsqueeze(-1)[index_mask]).detach() # detach from computation graph 
            rgb[index_mask] = rgb[index_mask] * fresnel_final

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
        return outputs
    
    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        return metrics_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        return loss_dict
    
    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics, images = super().get_image_metrics_and_images(outputs, batch)
        return metrics, images
