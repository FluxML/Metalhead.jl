# Building blocks shared by diffusers models (diffusers 0.42.0, Apache-2.0, The HuggingFace
# Team), from diffusers.models.embeddings, .normalization and .activations.

const ACTIVATIONS = Dict("silu" => swish, "swish" => swish)

function get_activation(name)
    haskey(ACTIVATIONS, name) ||
        error("activation `$(repr(name))` is not supported; supported: $(sort!(collect(keys(ACTIVATIONS))))")
    return ACTIVATIONS[name]
end

"""
    timestep_embedding(timesteps, embedding_dim; flip_sin_to_cos = false,
                       downscale_freq_shift = 1, scale = 1, max_period = 10_000)

diffusers' `get_timestep_embedding`: the sinusoidal embeddings of a vector of (possibly
fractional) timesteps, of size `(embedding_dim, length(timesteps))`, on the device of
`timesteps`. As in diffusers, they are computed in at least `Float32`: in `BFloat16`, the angles
of timesteps near 1000 would be off by several radians.
"""
function timestep_embedding(timesteps::AbstractVector, embedding_dim::Integer;
        flip_sin_to_cos::Bool = false, downscale_freq_shift = 1, scale = 1,
        max_period = 10_000)
    iseven(embedding_dim) ||
        error("`embedding_dim` ($embedding_dim) must be even in the Metalhead port")
    return ChainRulesCore.ignore_derivatives() do
        t = promote_type(Float32, eltype(timesteps)).(timesteps)
        half = embedding_dim ÷ 2
        exponents = [-log(max_period) * i / (half - downscale_freq_shift) for i in 0:(half - 1)]
        args = scale .* to_device_of(t, exp.(exponents)) .* transpose(t)
        return flip_sin_to_cos ? vcat(cos.(args), sin.(args)) : vcat(sin.(args), cos.(args))
    end
end

# diffusers' `Timesteps`, the parameter-free projection of the timesteps.
@concrete struct Timesteps
    num_channels
    flip_sin_to_cos
    downscale_freq_shift
    scale
end
Flux.@layer Timesteps

function (m::Timesteps)(t)
    return timestep_embedding(t, m.num_channels; m.flip_sin_to_cos, m.downscale_freq_shift,
        m.scale)
end

# diffusers' `TimestepEmbedding`, without the conditioning projection and the activation after
# the second layer, which are optional.
@concrete struct TimestepEmbedding
    linear_1
    act
    linear_2
end
Flux.@layer TimestepEmbedding

(m::TimestepEmbedding)(x) = m.linear_2(m.act(m.linear_1(x)))

function timestep_embedding_layer(in_channels, time_embed_dim; act_fn = "silu", bias = true)
    return TimestepEmbedding(Dense(in_channels => time_embed_dim; bias), get_activation(act_fn),
        Dense(time_embed_dim => time_embed_dim; bias))
end

"""
    AdaLayerNormContinuous(embedding_dim, conditioning_embedding_dim; elementwise_affine = true,
                           eps = 1e-5, bias = true)

diffusers' adaptive LayerNorm of tokens `(embedding_dim, seq_len, batch)`, scaled and shifted by
a projection of the conditioning embedding `(conditioning_embedding_dim, batch)`.
"""
@concrete struct AdaLayerNormContinuous
    silu
    linear
    norm
end
Flux.@layer AdaLayerNormContinuous

function AdaLayerNormContinuous(embedding_dim::Integer, conditioning_embedding_dim::Integer;
        elementwise_affine::Bool = true, eps = 1e-5, bias::Bool = true)
    return AdaLayerNormContinuous(swish,
        Dense(conditioning_embedding_dim => 2embedding_dim; bias),
        LayerNorm(embedding_dim; affine = elementwise_affine, eps))
end

function (m::AdaLayerNormContinuous)(x, conditioning_embedding)
    emb = m.linear(m.silu(conditioning_embedding))
    d = size(x, 1)
    scale = reshape(emb[1:d, :], d, 1, :)
    shift = reshape(emb[(d + 1):end, :], d, 1, :)
    return m.norm(x) .* (1 .+ scale) .+ shift
end

# A torch `nn.Linear` followed by an `nn.Dropout`, held in an `nn.ModuleList` (`to_out`).
linear_dropout(in_out, dropout; bias = true) = Chain(Dense(in_out; bias), Dropout(dropout))

# torch clips float16 hidden states to its finite range after some residual additions.
clip_float16(x) = eltype(x) == Float16 ? clamp.(x, -65504, 65504) : x
