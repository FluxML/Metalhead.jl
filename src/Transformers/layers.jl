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
the features divided by their root mean square, computed in at least `Float32`. Under
`f16`/`bf16` its `weight` stays `Float32`, and the output has the eltype of the input.
"""
@concrete struct RMSNorm
    weight
    variance_epsilon
end
Flux.@layer RMSNorm
Flux.f16(m::RMSNorm, cx) = f32(m, cx)
Flux.bf16(m::RMSNorm, cx) = f32(m, cx)

RMSNorm(dim::Integer; eps) = RMSNorm(ones(Float32, dim), Float64(eps))

(m::RMSNorm)(x) = NNlib.rmsnorm(m.weight, x; eps = m.variance_epsilon)

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

# The number of tokens of each sequence of a batch padded on the right, from a tokenizer's
# `attention_mask` of size `(seq_len, batch_size)`: the `kv_lengths` of
# `NNlib.scaled_dot_product_attention`, which masks the same keys as transformers.
function kv_lengths_of(attention_mask, input_ids)
    attention_mask === nothing && return nothing
    size(attention_mask) == size(input_ids) ||
        error("`attention_mask` has size $(size(attention_mask)), the token indices $(size(input_ids))")
    mask = Array(attention_mask) .!= 0
    lengths = vec(sum(mask; dims = 1))
    all(I -> mask[I] == (I[1] <= lengths[I[2]]), CartesianIndices(mask)) ||
        error("`attention_mask` must pad on the right: ones, then zeros, in each column")
    return lengths
end
ChainRulesCore.@non_differentiable kv_lengths_of(::Any...)

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

# transformers' `head_dim` and `num_key_value_heads` of an attention layer, from their
# config values, which may be `nothing`.
function attention_heads(; hidden_size, num_attention_heads, num_key_value_heads, head_dim)
    if head_dim === nothing
        hidden_size % num_attention_heads == 0 ||
            error("`hidden_size` ($hidden_size) must be divisible by `num_attention_heads` ($num_attention_heads)")
        head_dim = hidden_size ÷ num_attention_heads
    end
    num_key_value_heads = something(num_key_value_heads, num_attention_heads)
    num_attention_heads % num_key_value_heads == 0 ||
        error("`num_attention_heads` ($num_attention_heads) must be divisible by `num_key_value_heads` ($num_key_value_heads)")
    return head_dim, num_key_value_heads
end

# transformers' `layer_types` of a model with optional sliding-window attention (Qwen2, Qwen3),
# computed as its config's `__post_init__` does when the config has none. Only full attention
# is supported.
function check_full_attention(name; num_hidden_layers, use_sliding_window, sliding_window,
        max_window_layers, layer_types)
    window = use_sliding_window ? sliding_window : nothing
    types = something(layer_types,
        [window !== nothing && i >= max_window_layers ? "sliding_attention" :
         "full_attention"
         for i in 0:(num_hidden_layers - 1)])
    all(==("full_attention"), types) ||
        error("`layer_types = $(repr(Tuple(types)))` is not supported by the Metalhead port of $name: only full attention is")
    return nothing
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
