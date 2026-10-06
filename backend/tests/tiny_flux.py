"""A tiny randomly-initialised FluxFillPipeline (CPU, seconds) for testing Innovation A through the real
diffusers code path. No downloads."""
from __future__ import annotations

import torch


def tiny_fill_pipeline():
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxFillPipeline, FluxTransformer2DModel

    torch.manual_seed(0)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8), layers_per_block=1, latent_channels=4, norm_num_groups=4,
        use_quant_conv=False, use_post_quant_conv=False, shift_factor=0.0609, scaling_factor=1.5035,
    )
    # vae_scale_factor = 2 ** (4-1) = 8  (same as real FLUX); packed latent ch = 4*4 = 16,
    # masked-image latents 16 + packed mask 8*8*4 = 256  ->  in_channels = 16 + 16 + 256 = 288
    tr = FluxTransformer2DModel(
        patch_size=1, in_channels=288, out_channels=16, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=32,
        guidance_embeds=True, axes_dims_rope=[4, 4, 8],
    )
    sched = FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
                                            base_image_seq_len=256, max_image_seq_len=4096)
    pipe = FluxFillPipeline(scheduler=sched, vae=vae, transformer=tr, text_encoder=None, tokenizer=None,
                            text_encoder_2=None, tokenizer_2=None)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def tiny_embeds(L: int = 8):
    g = torch.Generator().manual_seed(1)
    return torch.randn(1, L, 32, generator=g), torch.randn(1, 32, generator=g)
