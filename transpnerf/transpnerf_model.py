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

@dataclass
class TranspNerfModelConfig(NerfactoModelConfig):
    """TranspNerf Model Configuration.
    """

    _target: Type = field(default_factory=lambda: TranspNerfModel)
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""


class TranspNerfModel(NerfactoModel):
    """TranspNerf Model."""

    config: TranspNerfModelConfig

    def populate_modules(self):
        super().populate_modules()
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.

    def _fresnel(self, n, in_dir, out_dir, normal):
        ## taken from https://github.com/dawning77/NeRRF

        in_dot = (in_dir * normal).sum(-1)
        out_dot = (out_dir * normal).sum(-1)

        F = ((in_dot - n * out_dot) / (in_dot + n * out_dot)) ** 2 + (
            (n * in_dot - out_dot) / (n * in_dot + out_dot)
        ) ** 2
        return F / 2
    
    def _reflection(self, ray_bundle: RayBundle, calc_fresnel: bool):
        #print("refl")
        depth = ray_bundle.metadata["depth"]
        normal = ray_bundle.metadata["normal"]
        input_origins = ray_bundle.origins
        input_directions = ray_bundle.directions
       
        # Generate a mask to exclude the background
        target_tensor = torch.tensor([0., 0., 0.]).cuda()  # target black normal (background)
        index_mask = torch.all(normal != target_tensor, dim=1) # indicies that are not background

        #index_mask = torch.all(depth != 0, dim=1)

        #print("index mask True count --> ", torch.sum(index_mask).item())
        # calculate incident angle
        cos_theta_i = (-input_directions * normal).sum(dim=1) 

        # calculate reflected rays origins and directions
        refl_origins = input_origins + depth.expand(-1, 3) * input_directions
        refl_dir = input_directions + 2 * cos_theta_i.unsqueeze(-1) * normal
        refl_dir = refl_dir / torch.norm(refl_dir, dim=-1).unsqueeze(-1)

        # new line
        #refl_origins = refl_origins - depth.expand(-1, 3) * refl_dir

        # use only reflected part for now
        ray_bundle.origins[index_mask] = refl_origins.clone()[index_mask]
        ray_bundle.directions[index_mask] = refl_dir.clone()[index_mask].to(ray_bundle.directions.dtype)


        #fresenel 
        if calc_fresnel:
            ior = 1.5
            ior_ = 1/ior
            cos_theta_o = torch.sqrt(1- ior**2 * (1- cos_theta_i**2))
            refract_dir_1 = -ior_*depth + (ior_*(-cos_theta_i) - cos_theta_o).unsqueeze(-1)*normal
            refract_dir_1 = refract_dir_1 / torch.norm(refract_dir_1, dim=-1).unsqueeze(-1)
            fresnel_1 = self._fresnel(ior, input_directions, refract_dir_1, normal)
            return ray_bundle, fresnel_1, index_mask


        return ray_bundle

    # def _refraction_w_reflect(self, ray_bundle: RayBundle, only_reflect: boolean):
    #     # only with the first intersection of the refractive surface

    #     ior = 1/1.5 #for now
    #     depth = ray_bundle.metadata["depth"]
    #     normal = ray_bundle.metadata["normal"]
    #     input_origins = ray_bundle.origins
    #     input_directions = ray_bundle.directions

    #     # Generate a mask to exclude the background
    #     target_tensor = torch.tensor([0., 0., 0.]).cuda()  # target black normal (background)
    #     index_mask = torch.all(normal != target_tensor, dim=1) # indicies that are not background
        
    #     # cos theta i 
    #     cos_theta_i = (-input_directions * normal).sum(dim=1) 

    #     # calculate reflected rays origins and directions
    #     refl_origins = input_origins + depth.expand(-1, 3) * input_directions
    #     refl_dir = input_directions + 2 * cos_theta_i.unsqueeze(-1) * normal
    #     refl_dir = refl_dir / torch.norm(refl_dir, dim=-1).unsqueeze(-1)

    #     if only_reflect:
    #         ray_bundle.origins[index_mask] = refl_origins.clone()[index_mask]
    #         ray_bundle.directions[index_mask] = refl_dir.clone()[index_mask].to(ray_bundle.directions.dtype)
    #         return ray_bundle

        
    #     # first refraction
    #     cos_theta_o = torch.sqrt(1- ior**2 * (1- cos_theta_i**2))
    #     refract_dir_1 = -ior*depth + (ior*(-cos_theta_i) - cos_theta_o).unsqueeze(-1)*normal
    #     refract_dir_1 = refract_dir_1 / torch.norm(refract_dir_1, dim=-1).unsqueeze(-1)
    #     refract_origins_1 = refl_origins

    #     # second refraction
    #     refract_origin_2 = refract_origins_1 + depth.expand(-1,3) * refract_dir_1
    #     cos_theta_i_2 = (-refract_dir_1 * normal).sum(dim=1)
    #     cos_theta_o_2 = torch.sqrt(1- ior**2 * (1- cos_theta_i_2**2)) #below zero might remove
    #     refract_dir_2 = 



        
    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        
        apply_refl = True
        apply_before = True
        calc_fresnel = False
        
        if apply_before and apply_refl:
            # apply reflection
            if "depth" in ray_bundle.metadata.keys() and "normal" in ray_bundle.metadata.keys():
                if calc_fresnel:
                    ray_bundle, fresnel_1, index_mask = self._reflection(ray_bundle, calc_fresnel)
                else:
                    ray_bundle = self._reflection(ray_bundle, calc_fresnel)
        
        # proposal sampler
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        
        
        if not(apply_before) and apply_refl:
            print("no")
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            ray_samples: RaySamples

            if "depth" in ray_bundle.metadata.keys() and "normal" in ray_bundle.metadata.keys():
                ray_bundle = self._reflection(ray_bundle)

            ray_samples =  self.sampler_pdf(ray_bundle, ray_samples, weights)
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
            if self.config.use_gradient_scaling:
                field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        
        # regular pdf way 
        # weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        # # pdf sampling
        # ray_samples = self.sampler_pdf(ray_bundle, ray_samples, weights)
        # field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        # if self.config.use_gradient_scaling:
        #     field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        if calc_fresnel:
            rgb[index_mask] = 1 * rgb[index_mask] #(1 - fresnel_1[index_mask].unsqueeze(-1)) * rgb[index_mask]


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
