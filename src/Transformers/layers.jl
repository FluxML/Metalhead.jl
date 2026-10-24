# Building blocks shared by the decoder-only language models of transformers
# (transformers 5.20, Apache-2.0, The Hugging Face Team).

# transformers' `_init_weights` of nn.Linear and nn.Embedding
normal_init(std) = (dims...) -> Float32(std) .* randn(Float32, dims...)

const ACT2FN = Dict("silu" => swish, "gelu_pytorch_tanh" => gelu_tanh)

function act_fn(name; key = "hidden_act")
    haskey(ACT2FN, name) ||
        error("`$key = $(repr(name))` is not supported; supported: $(sort!(collect(keys(ACT2FN))))")
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
    RotaryEmbedding(head_dim, rope_parameters)

transformers' rotary embedding, with the `rope_parameters` of `rope_parameters_of`: the
default one, or `rope_type = "linear"` or `"llama3"` (Llama 3.1 and later), which scale its
frequencies. Its `inv_freq` buffer is not persistent, so it has no `state_dict` entry: the
tables are computed on each call instead.
"""
@concrete struct RotaryEmbedding
    head_dim
    rope_parameters
end
Flux.@layer RotaryEmbedding

# transformers' `inv_freq`, in Float64. "linear" divides the default frequencies by
# `factor`. "llama3" divides by `factor` those whose wavelength is above
# `original_max_position_embeddings / low_freq_factor`, keeps those below
# `original_max_position_embeddings / high_freq_factor` and interpolates in between, as
# `_compute_llama3_parameters` does.
function inv_freq(p, head_dim)
    freqs = Float64(p.rope_theta) .^ (-(0:2:(head_dim - 2)) ./ head_dim)
    p.rope_type == "linear" && return freqs ./ p.factor
    p.rope_type == "llama3" || return freqs
    old_context_len = p.original_max_position_embeddings
    return map(freqs) do f
        wavelen = 2π / f
        wavelen < old_context_len / p.high_freq_factor && return f
        wavelen > old_context_len / p.low_freq_factor && return f / p.factor
        smooth = (old_context_len / wavelen - p.low_freq_factor) /
                 (p.high_freq_factor - p.low_freq_factor)
        return (1 - smooth) * f / p.factor + smooth * f
    end
end

# The tables of `NNlib.apply_rotary_embedding` for the positions 0:n-1, transformers'
# `position_embeddings`, computed in Float64 (transformers computes them in Float32) and
# returned in the element type and on the device of `x`. Its default pairing of the features
# k and k + head_dim ÷ 2 is transformers' `rotate_half`.
function rotary_tables(m::RotaryEmbedding, x, n)
    return ChainRulesCore.ignore_derivatives() do
        angles = inv_freq(m.rope_parameters, m.head_dim) .* (0:(n - 1))'
        T = eltype(x)
        return to_device_of(x, T.(cos.(angles))), to_device_of(x, T.(sin.(angles)))
    end
end

# The keys that each query attends to in sliding-window attention, on top of the causal
# mask: the `mask` of `NNlib.scaled_dot_product_attention`, of size `(seq_len, seq_len)`,
# true where the key `j` is in the window of the query `i`, `j > i - sliding_window` as in
# transformers' `sliding_window_overlay`. `nothing` where the window holds the whole
# sequence.
function sliding_window_mask(x, n, sliding_window)
    (sliding_window === nothing || sliding_window >= n) && return nothing
    return ChainRulesCore.ignore_derivatives() do
        mask = [j > i - sliding_window for j in 1:n, i in 1:n]
        return copyto!(similar(x, Bool, n, n), mask)
    end
end

# In a batch padded on the right, the queries past `kv_lengths + sliding_window - 1` have
# only padding in their window. PyTorch's attention gives them zeros, as
# `NNlib.scaled_dot_product_attention` does to the queries past its `q_lengths`, rather
# than NaNs.
function sliding_window_q_lengths(kv_lengths, n, sliding_window)
    (kv_lengths === nothing || sliding_window === nothing || sliding_window >= n) &&
        return nothing
    return min.(kv_lengths .+ (sliding_window - 1), n)
end
ChainRulesCore.@non_differentiable sliding_window_q_lengths(::Any...)

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
function rope_parameters_of(name; rope_theta, rope_scaling, rope_parameters,
        max_position_embeddings)
    return rope_type_parameters(name, something(rope_scaling, rope_parameters, (;));
        rope_theta = something(rope_theta, 10_000.0), max_position_embeddings)
end

# The parameters of a rotary embedding as a NamedTuple, from a dictionary or NamedTuple of
# transformers' `rope_parameters`: its `rope_type`, its `rope_theta` (the keyword where it
# has none) and the parameters of its scaling. `original_max_position_embeddings` defaults
# to `max_position_embeddings`, as in `standardize_rope_params`.
function rope_type_parameters(name, params; rope_theta, max_position_embeddings)
    params = (; (Symbol(k) => v for (k, v) in pairs(params))...)
    rope_type = get(params, :rope_type, get(params, :type, "default"))
    theta = Float64(get(params, :rope_theta, rope_theta))
    rope_type == "default" && return (; rope_type, rope_theta = theta)
    rope_type in ("linear", "llama3") ||
        error("`rope_type = $(repr(rope_type))` is not supported by the Metalhead port of $name; supported: \"default\", \"linear\" and \"llama3\"")
    get(params, :partial_rotary_factor, 1) == 1 ||
        error("`partial_rotary_factor` is not supported by the Metalhead port of $name")
    factor = Float64(params.factor)
    rope_type == "linear" && return (; rope_type, rope_theta = theta, factor)
    return (; rope_type, rope_theta = theta, factor,
        low_freq_factor = Float64(params.low_freq_factor),
        high_freq_factor = Float64(params.high_freq_factor),
        original_max_position_embeddings = get(params, :original_max_position_embeddings,
            max_position_embeddings))
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
