# Ported from diffusers.models.autoencoders.autoencoder_kl_flux2 (diffusers 0.42.0,
# Apache-2.0, The HuggingFace Team). Validated against test/fixtures/vae_flux2_* and
# test/references/black-forest-labs--FLUX.2-klein-4B--vae.

@concrete struct AutoencoderKLFlux2
    encoder
    decoder
    quant_conv
    post_quant_conv
    bn
end
Flux.@layer AutoencoderKLFlux2

"""
    encode(vae, x) -> DiagonalGaussianDistribution

The posterior of the latents of the images `x`, of size `(width, height, 3, batch)` with values
in [-1, 1]: [`mode`](@ref Metalhead.Diffusers.mode) gives the latents, of size
`(width ÷ f, height ÷ f, latent_channels, batch)` for the downsampling factor `f`.
"""
function encode(m::AutoencoderKLFlux2, x)
    h = m.encoder(x)
    return DiagonalGaussianDistribution(m.quant_conv === nothing ? h : m.quant_conv(h))
end

"""
    decode(vae, z)

The images, with values about [-1, 1], of the latents `z`.
"""
function decode(m::AutoencoderKLFlux2, z)
    return m.decoder(m.post_quant_conv === nothing ? z : m.post_quant_conv(z))
end

(m::AutoencoderKLFlux2)(x) = decode(m, mode(encode(m, x)))

"""
    autoencoder_kl_flux2(; latent_channels = 32, block_out_channels = (128, 256, 512, 512), kwargs...)

Port of diffusers' `AutoencoderKLFlux2`, the autoencoder of FLUX.2, built from the arguments
of its constructor (the `config.json` of the `vae` of a pipeline), with the same names and
defaults. Only `DownEncoderBlock2D` and `UpDecoderBlock2D` blocks are supported, which all
checkpoints use.

The model maps images of size `(width, height, in_channels, batch)` to images of the same size,
through the mean of the posterior: see [`encode`](@ref Metalhead.Diffusers.encode) and
[`decode`](@ref Metalhead.Diffusers.decode). Its `bn` holds the mean and the variance of the
latents in patches of `patch_size`, with which FLUX.2 pipelines normalise them; it takes no part
in the model's forward pass.
"""
function autoencoder_kl_flux2(; in_channels = 3, out_channels = 3,
        down_block_types = ntuple(_ -> "DownEncoderBlock2D", 4),
        up_block_types = ntuple(_ -> "UpDecoderBlock2D", 4),
        block_out_channels = (128, 256, 512, 512), decoder_block_out_channels = nothing,
        layers_per_block = 2, act_fn = "silu", latent_channels = 32, norm_num_groups = 32,
        sample_size = 1024, use_quant_conv = true, use_post_quant_conv = true,
        mid_block_add_attention = true, batch_norm_eps = 1e-4, batch_norm_momentum = 0.1,
        patch_size = (2, 2))
    encoder = vae_encoder(; in_channels, out_channels = latent_channels, down_block_types,
        block_out_channels, layers_per_block, act_fn, norm_num_groups, double_z = true,
        mid_block_add_attention)
    decoder = vae_decoder(; in_channels = latent_channels, out_channels, up_block_types,
        block_out_channels = something(decoder_block_out_channels, block_out_channels),
        layers_per_block, norm_num_groups, act_fn, mid_block_add_attention)
    quant_conv = use_quant_conv ? Conv((1, 1), 2latent_channels => 2latent_channels) : nothing
    post_quant_conv = use_post_quant_conv ? Conv((1, 1), latent_channels => latent_channels) :
                      nothing
    bn = BatchNorm(prod(patch_size) * latent_channels; affine = false, track_stats = true,
        eps = Float32(batch_norm_eps), momentum = Float32(batch_norm_momentum))
    return AutoencoderKLFlux2(encoder, decoder, quant_conv, post_quant_conv, bn)
end

register!(ArchSpec(DIFFUSERS, "AutoencoderKLFlux2", autoencoder_kl_flux2; family = "flux2",
    references = ["black-forest-labs/FLUX.2-klein-4B/vae" => "e7b7dc27f91deacad38e78976d1f2b499d76a294"]))
