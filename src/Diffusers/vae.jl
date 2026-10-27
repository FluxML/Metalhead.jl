# Ported from diffusers.models.autoencoders.vae, .resnet, .downsampling, .upsampling,
# .unets.unet_2d_blocks and the `Attention` of .attention_processor (diffusers 0.42.0,
# Apache-2.0, The HuggingFace Team): the encoder and decoder of the variational autoencoders,
# in the configurations they use (no time embedding, `DownEncoderBlock2D` and
# `UpDecoderBlock2D` blocks).

# diffusers' `ResnetBlock2D` without a time embedding (`temb_channels = None`).
@concrete struct ResnetBlock2D
    norm1
    conv1
    norm2
    dropout
    conv2
    nonlinearity
    conv_shortcut
end
Flux.@layer ResnetBlock2D

function resnet_block_2d(in_channels, out_channels; groups, eps, act_fn, dropout = 0.0)
    shortcut = in_channels == out_channels ? nothing :
               init_layer(Conv, (1, 1), in_channels => out_channels)
    return ResnetBlock2D(init_layer(GroupNorm, in_channels, groups; eps = Float32(eps)),
        init_layer(Conv, (3, 3), in_channels => out_channels; pad = 1),
        init_layer(GroupNorm, out_channels, groups; eps = Float32(eps)), Dropout(dropout),
        init_layer(Conv, (3, 3), out_channels => out_channels; pad = 1), get_activation(act_fn), shortcut)
end

function (m::ResnetBlock2D)(x)
    h = m.conv1(m.nonlinearity(m.norm1(x)))
    h = m.conv2(m.dropout(m.nonlinearity(m.norm2(h))))
    shortcut = m.conv_shortcut === nothing ? x : m.conv_shortcut(x)
    return shortcut .+ h
end

# diffusers' `Downsample2D` with a convolution: its `padding = 0` pads the right and the bottom
# by one, which the convolution does itself.
@concrete struct Downsample2D
    conv
end
Flux.@layer Downsample2D

(m::Downsample2D)(x) = m.conv(x)

# diffusers' `Upsample2D` with a convolution: nearest-neighbour upsampling by 2, then the conv.
@concrete struct Upsample2D
    conv
end
Flux.@layer Upsample2D

(m::Upsample2D)(x) = m.conv(NNlib.upsample_nearest(x, (2, 2)))

# diffusers' `Attention` as the mid blocks of the autoencoders build it: self-attention of the
# positions of a WHCN feature map, after a GroupNorm, with a residual connection.
@concrete struct Attention
    heads
    group_norm
    to_q
    to_k
    to_v
    to_out
end
Flux.@layer Attention

function vae_attention(channels; groups, eps)
    return Attention(1, init_layer(GroupNorm, channels, groups; eps = Float32(eps)),
        init_layer(Dense, channels => channels), init_layer(Dense, channels => channels), init_layer(
            Dense, channels => channels),
        linear_dropout(channels => channels, 0.0))
end

function (m::Attention)(x)
    w, h, c, b = size(x)
    tokens = permutedims(reshape(m.group_norm(x), w * h, c, b), (2, 1, 3))
    heads(y) = reshape(y, :, m.heads, w * h, b)
    y = NNlib.scaled_dot_product_attention(heads(m.to_q(tokens)), heads(m.to_k(tokens)),
        heads(m.to_v(tokens)))
    y = m.to_out(reshape(y, :, w * h, b))
    return reshape(permutedims(y, (2, 1, 3)), w, h, c, b) .+ x
end

@concrete struct UNetMidBlock2D
    attentions
    resnets
end
Flux.@layer UNetMidBlock2D

# A Tuple, not a Vector: `gpu` would turn `[nothing]` into a GPU array.
function unet_mid_block_2d(channels; groups, eps, act_fn, add_attention)
    attentions = (add_attention ? vae_attention(channels; groups, eps) : nothing,)
    resnets = [resnet_block_2d(channels, channels; groups, eps, act_fn) for _ in 1:2]
    return UNetMidBlock2D(attentions, resnets)
end

function (m::UNetMidBlock2D)(x)
    x = m.resnets[1](x)
    for (attn, resnet) in zip(m.attentions, m.resnets[2:end])
        x = attn === nothing ? x : attn(x)
        x = resnet(x)
    end
    return x
end

@concrete struct DownEncoderBlock2D
    resnets
    downsamplers
end
Flux.@layer DownEncoderBlock2D

function down_encoder_block_2d(in_channels, out_channels; num_layers, add_downsample, groups,
        eps, act_fn)
    resnets = [resnet_block_2d(i == 1 ? in_channels : out_channels, out_channels; groups, eps,
                   act_fn) for i in 1:num_layers]
    down = init_layer(
        Conv, (3, 3), out_channels => out_channels; stride = 2, pad = (0, 1, 0, 1))
    return DownEncoderBlock2D(resnets, add_downsample ? [Downsample2D(down)] : nothing)
end

function (m::DownEncoderBlock2D)(x)
    for resnet in m.resnets
        x = resnet(x)
    end
    m.downsamplers === nothing && return x
    for downsampler in m.downsamplers
        x = downsampler(x)
    end
    return x
end

@concrete struct UpDecoderBlock2D
    resnets
    upsamplers
end
Flux.@layer UpDecoderBlock2D

function up_decoder_block_2d(in_channels, out_channels; num_layers, add_upsample, groups, eps,
        act_fn)
    resnets = [resnet_block_2d(i == 1 ? in_channels : out_channels, out_channels; groups, eps,
                   act_fn) for i in 1:num_layers]
    up = init_layer(Conv, (3, 3), out_channels => out_channels; pad = 1)
    return UpDecoderBlock2D(resnets, add_upsample ? [Upsample2D(up)] : nothing)
end

function (m::UpDecoderBlock2D)(x)
    for resnet in m.resnets
        x = resnet(x)
    end
    m.upsamplers === nothing && return x
    for upsampler in m.upsamplers
        x = upsampler(x)
    end
    return x
end

function check_block_types(name, types, expected)
    all(==(expected), types) ||
        error("`$name = $(repr(types))` is not supported by the Metalhead port: only $expected blocks are")
    return nothing
end

@concrete struct Encoder
    conv_in
    down_blocks
    mid_block
    conv_norm_out
    conv_act
    conv_out
end
Flux.@layer Encoder

function vae_encoder(; in_channels, out_channels, down_block_types, block_out_channels,
        layers_per_block, norm_num_groups, act_fn, double_z = true, mid_block_add_attention)
    check_block_types("down_block_types", down_block_types, "DownEncoderBlock2D")
    length(down_block_types) == length(block_out_channels) ||
        error("`down_block_types` and `block_out_channels` must have the same length")
    kw = (; groups = norm_num_groups, eps = 1e-6, act_fn)
    n = length(block_out_channels)
    down_blocks = [down_encoder_block_2d(block_out_channels[max(i - 1, 1)], block_out_channels[i];
                       num_layers = layers_per_block, add_downsample = i < n, kw...)
                   for i in 1:n]
    channels = block_out_channels[end]
    return Encoder(init_layer(Conv, (3, 3), in_channels => block_out_channels[1]; pad = 1),
        down_blocks,
        unet_mid_block_2d(channels; kw..., add_attention = mid_block_add_attention),
        init_layer(GroupNorm, channels, norm_num_groups; eps = 1.0f-6), get_activation(act_fn),
        init_layer(Conv, (3, 3), channels => (double_z ? 2 : 1) * out_channels; pad = 1))
end

function (m::Encoder)(x)
    x = m.conv_in(x)
    for block in m.down_blocks
        x = block(x)
    end
    x = m.mid_block(x)
    return m.conv_out(m.conv_act(m.conv_norm_out(x)))
end

@concrete struct Decoder
    conv_in
    up_blocks
    mid_block
    conv_norm_out
    conv_act
    conv_out
end
Flux.@layer Decoder

function vae_decoder(; in_channels, out_channels, up_block_types, block_out_channels,
        layers_per_block, norm_num_groups, act_fn, mid_block_add_attention)
    check_block_types("up_block_types", up_block_types, "UpDecoderBlock2D")
    length(up_block_types) == length(block_out_channels) ||
        error("`up_block_types` and `block_out_channels` must have the same length")
    kw = (; groups = norm_num_groups, eps = 1e-6, act_fn)
    reversed = reverse(collect(block_out_channels))
    n = length(reversed)
    up_blocks = [up_decoder_block_2d(reversed[max(i - 1, 1)], reversed[i];
                     num_layers = layers_per_block + 1, add_upsample = i < n, kw...)
                 for i in 1:n]
    return Decoder(
        init_layer(Conv, (3, 3), in_channels => reversed[1]; pad = 1), up_blocks,
        unet_mid_block_2d(reversed[1]; kw..., add_attention = mid_block_add_attention),
        init_layer(GroupNorm, block_out_channels[1], norm_num_groups; eps = 1.0f-6), get_activation(act_fn),
        init_layer(Conv, (3, 3), block_out_channels[1] => out_channels; pad = 1))
end

function (m::Decoder)(z)
    x = m.mid_block(m.conv_in(z))
    for block in m.up_blocks
        x = block(x)
    end
    return m.conv_out(m.conv_act(m.conv_norm_out(x)))
end

"""
    DiagonalGaussianDistribution(parameters)

diffusers' posterior of an autoencoder: a Gaussian with the mean and the log-variance (clamped
to [-30, 20]) that are the two halves of the channels of the encoder's output `parameters`.
[`mode`](@ref Metalhead.Diffusers.mode) returns its mean, and
[`sample`](@ref Metalhead.Diffusers.sample) draws from it.
"""
struct DiagonalGaussianDistribution{A}
    mean::A
    logvar::A
end

function DiagonalGaussianDistribution(parameters::AbstractArray{<:Any, 4})
    c = size(parameters, 3) ÷ 2
    return DiagonalGaussianDistribution(parameters[:, :, 1:c, :],
        clamp.(parameters[:, :, (c + 1):end, :], -30, 20))
end

"""
    mode(posterior::DiagonalGaussianDistribution)

The mean of the posterior of an autoencoder.
"""
mode(d::DiagonalGaussianDistribution) = d.mean

"""
    sample([rng], posterior::DiagonalGaussianDistribution)

A draw from the posterior of an autoencoder, with noise from `rng` (generated on the CPU).
"""
function sample(rng::AbstractRNG, d::DiagonalGaussianDistribution)
    noise = ChainRulesCore.ignore_derivatives() do
        to_device_of(d.mean, randn(rng, Float32, size(d.mean)))
    end
    return d.mean .+ exp.(d.logvar ./ 2) .* noise
end
sample(d::DiagonalGaussianDistribution) = sample(Random.default_rng(), d)
