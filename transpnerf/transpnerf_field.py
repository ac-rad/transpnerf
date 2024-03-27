# """
# TranspNerf Nerfstudio Field

# Note: this is only used with the calc_fresnel option.
# """

from typing import Dict, Literal, Optional, Tuple
from typing import Dict, Tuple
from torch import Tensor, nn
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.field_components.activations import trunc_exp
import torch
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion

class TranspNerfField(NerfactoField):
    """TranspNerf Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        average_init_density: float = 1.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__(aabb=aabb, num_images=num_images)

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        if self.appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.embedding_appearance = None
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients
        self.base_res = base_res
        self.average_init_density = average_init_density
        self.step = 0

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2 - 1, implementation=implementation
        )

        self.mlp_base = MLPWithHashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        # transients
        if self.use_transient_embedding:
            self.transient_embedding_dim = transient_embedding_dim
            self.embedding_transient = Embedding(self.num_images, self.transient_embedding_dim)
            self.mlp_transient = MLP(
                in_dim=self.geo_feat_dim + self.transient_embedding_dim,
                num_layers=num_layers_transient,
                layer_width=hidden_dim_transient,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_transient_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_transient.get_out_dim())
            self.field_head_transient_rgb = TransientRGBFieldHead(in_dim=self.mlp_transient.get_out_dim())
            self.field_head_transient_density = TransientDensityFieldHead(in_dim=self.mlp_transient.get_out_dim())

        # semantics
        if self.use_semantics:
            self.mlp_semantics = MLP(
                in_dim=self.geo_feat_dim,
                num_layers=2,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.get_out_dim(), num_classes=num_semantic_classes
            )

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = MLP(
                in_dim=self.geo_feat_dim + self.position_encoding.get_out_dim(),
                num_layers=3,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.get_out_dim())

        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        density, base_mlp_out = super().get_density(ray_samples)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None, fresnel_info: Optional[Dict] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        embedded_appearance = None
        if self.embedding_appearance is not None:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
            ]
            + (
                [embedded_appearance.view(-1, self.appearance_embedding_dim)] if embedded_appearance is not None else []
            ),
            dim=-1,
        )

        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)

        if fresnel_info and "index_mask" in fresnel_info:
            index_mask = fresnel_info["index_mask"]
            fresnel_final = 1 - fresnel_info["fresnel"].unsqueeze(1).unsqueeze(2).expand(-1, 48, 3)[index_mask].clone().detach()
            new_rgb = (rgb[index_mask].clone() * fresnel_final).to(torch.float16) #.detach()
            rgb_clone = rgb.clone()
            rgb_clone[index_mask] = new_rgb
            rgb = rgb_clone

        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
    
    def forward(self, ray_samples: RaySamples, compute_normals: bool = False, fresnel_info: Optional[Dict] = None) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """

        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding, fresnel_info=fresnel_info)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
