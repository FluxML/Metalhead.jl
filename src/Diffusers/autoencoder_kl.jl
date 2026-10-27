# Ported from diffusers.models.autoencoders.autoencoder_kl (diffusers 0.42.0, Apache-2.0,
# The HuggingFace Team), without tiling and slicing. Validated against
# test/fixtures/autoencoder_kl_* and test/references/Tongyi-MAI--Z-Image-Turbo--vae.

@concrete struct AutoencoderKL
    encoder
    decoder
    quant_conv
    post_quant_conv
    scaling_factor
    shift_factor
end
Flux.@layer AutoencoderKL

function encode(m::AutoencoderKL, x)
    h = m.encoder(x)
    return DiagonalGaussianDistribution(m.quant_conv === nothing ? h : m.quant_conv(h))
end

function decode(m::AutoencoderKL, z)
    return m.decoder(m.post_quant_conv === nothing ? z : m.post_quant_conv(z))
end

(m::AutoencoderKL)(x) = decode(m, mode(encode(m, x)))

"""
    autoencoder_kl(; latent_channels = 4, block_out_channels = (64,),
                   scaling_factor = 0.18215, shift_factor = nothing, kwargs...)

Port of diffusers' `AutoencoderKL`, the variational autoencoder of Stable Diffusion, FLUX.1
and Z-Image, built from the arguments of its constructor (the `config.json` of the `vae` of
a pipeline), with the same names and defaults. Only `DownEncoderBlock2D` and
`UpDecoderBlock2D` blocks are supported, which all checkpoints use.

The model maps images of size `(width, height, in_channels, batch)` to images of the same
size, through the mean of the posterior: see [`encode`](@ref Metalhead.Diffusers.encode) and
[`decode`](@ref Metalhead.Diffusers.decode). Pipelines decode `z / scaling_factor .+
shift_factor` from the latents `z` they denoise, which the model holds as its fields
`scaling_factor` and `shift_factor` (`nothing` for none). `sample_size`, `latents_mean`,
`latents_std` and `force_upcast` are settings of tiling and of other pipelines, which the
port ignores.
"""
function autoencoder_kl(; in_channels = 3, out_channels = 3,
        down_block_types = ("DownEncoderBlock2D",), up_block_types = ("UpDecoderBlock2D",),
        block_out_channels = (64,), layers_per_block = 1, act_fn = "silu",
        latent_channels = 4, norm_num_groups = 32, sample_size = 32,
        scaling_factor = 0.18215, shift_factor = nothing, latents_mean = nothing,
        latents_std = nothing, force_upcast = true, use_quant_conv = true,
        use_post_quant_conv = true, mid_block_add_attention = true)
    encoder = vae_encoder(; in_channels, out_channels = latent_channels, down_block_types,
        block_out_channels, layers_per_block, act_fn, norm_num_groups, double_z = true,
        mid_block_add_attention)
    decoder = vae_decoder(; in_channels = latent_channels, out_channels, up_block_types,
        block_out_channels, layers_per_block, norm_num_groups, act_fn,
        mid_block_add_attention)
    conv(c) = Conv((1, 1), c => c)
    quant_conv = use_quant_conv ? conv(2latent_channels) : nothing
    post_quant_conv = use_post_quant_conv ? conv(latent_channels) : nothing
    return AutoencoderKL(encoder, decoder, quant_conv, post_quant_conv,
        Float64(scaling_factor), shift_factor === nothing ? nothing : Float64(shift_factor))
end

register!(ArchSpec(DIFFUSERS, "AutoencoderKL", autoencoder_kl; family = "z_image",
    references = ["Tongyi-MAI/Z-Image-Turbo/vae" => "f332072aa78be7aecdf3ee76d5c247082da564a6"]))
