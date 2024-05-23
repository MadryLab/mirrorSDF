from typing import TYPE_CHECKING, Dict

import torch as ch
import torch.nn.functional as F

from ..architectures import MLPWithSkip
from ...utils.light_simulation import evaluate_samples_ray
from ...utils.materials import impute_L
from ...utils.sampling import equi_spaced_samples, sample_according_to_density

if TYPE_CHECKING:
    from ..embeddings import RandomFourierEmbedding
    from ...material_models.base import BaseMaterialModel
    from ...dataset.batch import Batch


class PBRNeRF(ch.nn.Module):

    def __init__(self, coord_encoder: 'RandomFourierEmbedding',
                 material_model: 'BaseMaterialModel',
                 pbr_material_model: 'BaseMaterialModel',
                 train_density_noise: float = 0.0,
                 coarse_width: int = 128,
                 coarse_n_layers: int = 4,
                 material_net_width: int = 128,
                 material_net_n_layers: int = 4,
                 **base_model_args):
        super().__init__()
        self.coord_encoder = coord_encoder
        self.material_model = material_model
        self.pbr_material_model = pbr_material_model
        self.train_density_noise = train_density_noise
        num_material_params = self.material_model.num_parameters
        # We need outputs for:
        # - 1 for Density
        # - 3 For the normal
        # - Whatever the material model requires
        self.output_param_counts = [
            1,  # Density
            3,  # Predicted gradient
            num_material_params
        ]
        self.inner = MLPWithSkip(coord_encoder.output_features + 3,
                                 sum(self.output_param_counts),
                                 **base_model_args)
        self.inner_coarse = MLPWithSkip(coord_encoder.output_features + 3,
                                        1,
                                        n_layers=4,
                                        width=128)

        self.material_params_predictor = MLPWithSkip(
            coord_encoder.output_features + 3 + num_material_params,
            self.pbr_material_model.num_parameters,
            n_layers=material_net_n_layers,
            width=material_net_width
        )

    def forward_coarse(self, coords: ch.Tensor):
        features = self.coord_encoder(coords)
        full_inputs = ch.concat([coords, features], dim=-1)

        density = self.inner_coarse(full_inputs)
        density = F.softplus(density) / 2
        return density

    def forward(self, coords: ch.Tensor):
        features = self.coord_encoder(coords)
        full_inputs = ch.concat([coords, features], dim=-1)

        prediction = self.inner(full_inputs)

        density, predicted_gradients, model_params = ch.split(prediction, self.output_param_counts, dim=-1)
        if self.training:
            density = density + ch.randn_like(density) * self.train_density_noise
        density = F.softplus(density) / 2

        return density, predicted_gradients, model_params, full_inputs

    def numerical_surface_gradient(self, coords: ch.Tensor, h: float, n_taps: int,
                                   compute_non_linearity: bool = False,
                                   randomize_lengths: bool = True):
        with ch.no_grad():
            random_directions = ch.randn(n_taps, 3, device=coords.device)
            if not randomize_lengths:
                random_directions = F.normalize(random_directions, p=2, dim=-1)
            else:
                h /= 1.57  # Take into account the average norm a 3D gaussian point
            random_directions *= h
            inv_dir = ch.linalg.pinv(random_directions).T.clone() * (h / 2)
            norm_directions = random_directions.norm(p=2, dim=-1, keepdim=True)
            inv_dir = inv_dir / norm_directions
            p_f = (coords[:, None] + random_directions[None])
            p_b = (coords[:, None] - random_directions[None])

        vf = self(p_f.reshape(-1, 3))[0].reshape(-1, n_taps)
        vb = self(p_b.reshape(-1, 3))[0].reshape(-1, n_taps)
        gradient = (vf - vb) @ inv_dir

        with ch.no_grad():
            gn = F.normalize(gradient, p=2, dim=-1).detach()
        if gradient.requires_grad:
            gradient.register_hook(lambda g: g - ch.sum(g * gn, dim=-1, keepdim=True) * gn)

        if compute_non_linearity:
            with ch.no_grad():
                center_values = (vf + vb) / 2
                projected_gradient = (random_directions[None] * gradient[:, None]).sum(-1)
                first_order_approx_forward = center_values + projected_gradient
                first_order_approx_backward = center_values - projected_gradient
            non_linearity2 = (
                    (first_order_approx_forward - vf) ** 2
                    + (first_order_approx_backward - vb) ** 2
            )
            return gradient, non_linearity2
        else:
            return gradient

    @staticmethod
    def compute_nerf_weights(t: ch.Tensor, densities: ch.Tensor, max_dist: ch.Tensor) -> ch.Tensor:
        t = ch.cat([t, max_dist], dim=1)
        deltas = t.diff(dim=1)
        prods = densities * deltas
        transmittance = ch.exp(-prods.cumsum(-1))
        shifted_transmittance = ch.nn.functional.pad(
            transmittance[:, :-1],
            (1, 0, 0, 0), mode="constant", value=1.0)
        individual_factors = (1 - ch.exp(-prods))
        result = shifted_transmittance * individual_factors
        return result.clip(0, 1)

    def render(self, batch: 'Batch', num_samples_corase: int, num_samples_fine: int,
               with_predicted_normals: bool = False,
               n_taps: int = 0, h: float = 1e-3,
               with_material_params: bool = False,
               pbr_mode: bool = False) -> Dict[str, ch.Tensor]:

        with ch.no_grad():
            normalized_t = equi_spaced_samples(batch['size'], num_samples_corase, batch['device'],
                                               randomized=self.training)
            locations, directions, t = evaluate_samples_ray(batch, normalized_t, foreground=True)
            locations_flat = locations.reshape(-1, 3)

        density_coarse = self.forward_coarse(locations_flat)
        density_coarse = density_coarse.reshape(batch['size'], num_samples_corase)
        with ch.no_grad():
            weights = PBRNeRF.compute_nerf_weights(t, density_coarse.reshape(t.shape) / 2, batch['foreground_t_range'][1])
            scaled_weights = F.normalize(weights.clip(1e-7), dim=-1, p=1)
            fine_t = sample_according_to_density(normalized_t, scaled_weights, num_samples_fine,
                                                 randomized=self.training)

            normalized_t, order = ch.sort(ch.cat([normalized_t, fine_t], dim=-1), dim=-1)
            inv_order = ch.empty_like(order)
            inv_order.scatter_(1, order, ch.arange(order.shape[1], device=order.device).expand(order.shape[0], -1))
            num_ray_samples = num_samples_corase + num_samples_fine

            locations, directions, t = evaluate_samples_ray(batch, normalized_t, foreground=True)
            locations_flat = locations.reshape(-1, 3)

        density, predicted_normals, material_params, full_inputs = self(locations_flat)
        fine_density_at_coarse_locs = ch.gather(density.reshape(t.shape), 1, inv_order[:, :num_samples_corase])

        env_ids = batch['env_id'][:, None].expand(-1, num_ray_samples).ravel()
        view_dir = directions.reshape(-1, 3)
        coords = locations.reshape(-1, 3)

        unit_normals = F.normalize(predicted_normals, p=2, dim=-1)
        rgb_linear = self.material_model(coords, env_ids,
                                         unit_normals, view_dir,
                                         material_params, batch)

        rgb_linear = rgb_linear.reshape(batch['size'], num_ray_samples, 3)
        predicted_normals = predicted_normals.reshape(batch['size'], num_ray_samples, 3)
        material_params = material_params.reshape(batch['size'], num_ray_samples, -1)
        weights = PBRNeRF.compute_nerf_weights(t, density.reshape(t.shape), batch['foreground_t_range'][1])
        fg_color = (rgb_linear * weights[..., None]).sum(1)
        fg_alpha = weights.sum(1)
        final_color = fg_color + batch['precomputed_background'] * (1 - fg_alpha)[..., None]
        result = {
            'normalized_t': normalized_t,
            'rgb': final_color,
            't': t,
            'density': density.reshape(t.shape),
            'weights': weights,
            'individual_rgb': rgb_linear,
            'all_coords': locations_flat,
            'all_p_normals': predicted_normals.reshape(-1, 3),
            'view_dir': view_dir,
            'coarse_fine_diff': (smooth_coefs(fine_density_at_coarse_locs.detach()), density_coarse)
        }

        if with_predicted_normals:
            result['predicted_normals'] = (predicted_normals * weights[..., None]).sum(1)
            result['light_dir'] = (
                    impute_L(-view_dir.reshape(batch['size'], num_ray_samples, 3),
                             predicted_normals)
                    * weights[..., None]).sum(1)

        if n_taps > 0:
            result['numerical_gradients'] = (predicted_normals * weights[..., None]).sum(1)

        if n_taps > 0:
            numerical_gradients = self.numerical_surface_gradient(locations_flat, h, n_taps,
                                                                  compute_non_linearity=False,
                                                                  randomize_lengths=self.training)
            numerical_gradients = numerical_gradients.reshape(batch['size'], num_ray_samples, 3)
            numerical_normals = F.normalize(-numerical_gradients, p=2, dim=-1)
            weighted_grads = (numerical_gradients * weights[..., None]).sum(1)
            weighted_normals = (numerical_normals * weights[..., None]).sum(1)
            result['numerical_gradients'] = weighted_grads
            result['numerical_normals'] = weighted_normals

        if with_material_params:
            result['material_params'] = (material_params * weights[..., None]).sum(1)

        return result

    def normal_regularizations(self, render_result, num_reg_points: int, n_taps: int, h: float):
        sampling_probabilities = render_result['weights'].ravel().clip(1e-5)
        selected_indices = ch.multinomial(sampling_probabilities, num_reg_points, replacement=True)
        selected_locations = render_result['all_coords'].reshape(-1, 3)[selected_indices]
        selected_pred_normals = render_result['all_p_normals'][selected_indices]
        selected_true_gradients = self.numerical_surface_gradient(selected_locations,
                                                                  n_taps=n_taps, h=h,
                                                                  randomize_lengths=self.training)
        selected_true_normals = F.normalize(-selected_true_gradients, p=2, dim=-1)
        normal_prediction_loss = F.mse_loss(selected_pred_normals, selected_true_normals.detach())
        true_normals_loss = F.mse_loss(selected_pred_normals.detach(), selected_true_normals)
        return true_normals_loss, normal_prediction_loss

def smooth_coefs(coefs):
    coefs = F.pad(coefs, (1, 1), value=0.0)
    return 0.25 * coefs[:, 0:-2] + 0.5 * coefs[:, 1:-1] + 0.25 * coefs[:, 2:]

class PBRNeRFWrapper(ch.nn.Module):

    def __init__(self, pbr_nerf: 'PBRNeRF'):
        super().__init__()
        self.inner = pbr_nerf

    def forward(self, batch: 'Batch', num_samples_corase: int, num_samples_fine: int,
                n_taps: int = 0, h: float = 1e-3, num_regularization_points: int = 2048,
                ) -> Dict[str, ch.Tensor]:
        predictions = self.inner.render(batch, num_samples_corase, num_samples_fine)
        if num_regularization_points > 0:
            true_normal_loss, pred_normal_loss = self.inner.normal_regularizations(
                predictions, num_regularization_points, n_taps=n_taps, h=h)
            predictions['true_normal_loss'] = true_normal_loss
            predictions['pred_normal_loss'] = pred_normal_loss

        return predictions
