from typing import TypedDict, Optional, Tuple

import torch as ch


class Batch(TypedDict):
    device: ch.device
    size: int
    cx: ch.IntTensor
    cy: ch.IntTensor

    indices: ch.IntTensor
    env_id: ch.IntTensor

    origin: ch.Tensor
    direction: ch.Tensor

    mirror_vertices: ch.Tensor
    mirror_normal: ch.Tensor
    mirror_thickness: float
    eta: float
    offset_lut: ch.Tensor

    log_measurement: ch.Tensor
    tonemapped_measurement: ch.Tensor

    object_bounding_sphere_radius: ch.Tensor
    background_radius_range: Tuple[float, float]

    exit_bounce_coord: Optional[ch.Tensor]
    bounce_direction: Optional[ch.Tensor]
    enter_bounce_coord: Optional[ch.Tensor]
    t_bounce: Optional[ch.Tensor]

    background_t_range: Optional[Tuple[ch.Tensor, ch.Tensor]]
    foreground_t_range: Optional[Tuple[ch.Tensor, ch.Tensor]]

    precomputed_background: Optional[ch.Tensor]
