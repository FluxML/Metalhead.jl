# Building blocks shared by the decoder-only language models of transformers
# (transformers 5.20, Apache-2.0, The Hugging Face Team).

# transformers' `_init_weights` of nn.Linear and nn.Embedding
normal_init(std) = (dims...) -> Float32(std) .* randn(Float32, dims...)

const ACT2FN = Dict("silu" => swish)

function act_fn(name)
    haskey(ACT2FN, name) ||
        error("`hidden_act = $(repr(name))` is not supported; supported: $(sort!(collect(keys(ACT2FN))))")
    return ACT2FN[name]
end

"""
    RMSNorm(dim; eps)

transformers' `LlamaRMSNorm`, which other models copy under their own names: `weight` times
the features divided by their root mean square, computed in at least `Float32`.
"""
@concrete struct RMSNorm
    weight
    variance_epsilon
end
Flux.@layer RMSNorm

RMSNorm(dim::Integer; eps) = RMSNorm(ones(Float32, dim), Float64(eps))

(m::RMSNorm)(x) = m.weight .* NNlib.rmsnorm(nothing, x; eps = m.variance_epsilon)

"""
    RotaryEmbedding(head_dim, rope_theta)

transformers' rotary embedding with `rope_type = "default"`. Its `inv_freq` buffer is not
persistent, so it has no `state_dict` entry: the tables are computed on each call instead.
"""
@concrete struct RotaryEmbedding
    head_dim
    rope_theta
end
Flux.@layer RotaryEmbedding

# The tables of `NNlib.apply_rotary_embedding` for the positions 0:n-1, computed in Float64
# (transformers computes them in Float32) and returned in the element type and on the device
# of `x`. Its default pairing of the features k and k + head_dim ÷ 2 is transformers'
# `rotate_half`.
function rotary_tables(m::RotaryEmbedding, x, n)
    return ChainRulesCore.ignore_derivatives() do
        c, s = NNlib.rotary_embedding_table(m.head_dim, n, eltype(x); base = m.rope_theta)
        return to_device_of(x, c), to_device_of(x, s)
    end
end

to_device_of(x, a) = copyto!(similar(x, size(a)), a)

# transformers' `rope_parameters`, from it or from the older `rope_theta` and `rope_scaling`
# keys of config.json, as in `RotaryEmbeddingConfigMixin.convert_rope_params_to_dict`.
function rope_theta_of(name; rope_theta, rope_scaling, rope_parameters)
    params = something(rope_scaling, rope_parameters, (;))
    params = params isa AbstractDict ? (; (Symbol(k) => v for (k, v) in params)...) : params
    rope_type = get(params, :rope_type, get(params, :type, "default"))
    rope_type == "default" ||
        error("`rope_type = $(repr(rope_type))` is not supported by the Metalhead port of $name")
    return Float64(get(params, :rope_theta, something(rope_theta, 10_000.0)))
end

"""
    TiedLMHead(weight)

The output projection `lm_head` of a language model with `tie_word_embeddings`: `weight` is the
`(hidden, vocab)` matrix of the input `Embedding` itself, so the two stay tied. Its key
`lm_head.weight` is an alias, which a checkpoint may omit.
"""
@concrete struct TiedLMHead
    weight
end
Flux.@layer TiedLMHead

layer_params(l::TiedLMHead) = [("weight", l.weight, :plain)]

function (l::TiedLMHead)(x)
    y = transpose(l.weight) * reshape(x, size(x, 1), :)
    return reshape(y, :, size(x)[2:end]...)
end
