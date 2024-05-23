import torch as ch

sobol_engine_2d = ch.quasirandom.SobolEngine(2, scramble=True)


class UniformSampler:

    def sample(self, batch_size: int, num_samples: int, device: ch.device) -> ch.Tensor:
        u = ch.rand(batch_size, num_samples, 2, device=device)
        z = u[..., 0]
        r = (1 - z ** 2) ** 0.5
        phi = 2 * ch.pi * u[..., 1]
        return ch.stack([r * ch.cos(phi), r * ch.sin(phi), z], dim=-1)

    @staticmethod
    def pdf(L: ch.Tensor) -> ch.Tensor:
        return ch.full((L.shape[0], L.shape[1]), 1 / (2 * ch.pi), device=L.device)
