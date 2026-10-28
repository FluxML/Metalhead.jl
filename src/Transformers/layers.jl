# Building blocks shared by the decoder-only language models of transformers
# (transformers 5.20, Apache-2.0, The Hugging Face Team).

# transformers' `_init_weights` of nn.Linear and nn.Embedding
function normal_init(std)
    return (dims...) -> init_array((ds...) -> Float32(std) .* randn(Float32, ds...), dims...)
end

const ACT2FN = Dict("silu" => swish, "gelu_pytorch_tanh" => gelu_tanh)

function act_fn(name; key = "hidden_act", table = ACT2FN)
    haskey(table, name) ||
        error("`$key = $(repr(name))` is not supported; supported: $(sort!(collect(keys(table))))")
    return table[name]
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

function RMSNorm(dim::Integer; eps)
    return RMSNorm(init_array(Flux.ones32, dim; norm = true), Float64(eps))
end

(m::RMSNorm)(x) = NNlib.rmsnorm(m.weight, x; eps = m.variance_epsilon)

"""
    RotaryEmbedding(head_dim, rope_parameters)

transformers' rotary embedding, with the `rope_parameters` of `rope_parameters_of`: the
default one, or `rope_type = "linear"` or `"llama3"` (Llama 3.1 and later), which scale its
frequencies. Its `inv_freq` buffer is not persistent, so it has no `state_dict` entry. The
tables are computed on the input's device.
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
function inv_freq(p, head_dim, x)
    freqs = similar(x, Float64, head_dim ÷ 2)
    freqs .= Float64(p.rope_theta) .^ (.-(0:2:(head_dim - 2)) ./ head_dim)
    p.rope_type == "linear" && return freqs ./ p.factor
    p.rope_type == "llama3" || return freqs
    old_context_len = p.original_max_position_embeddings
    factor, low, high = p.factor, p.low_freq_factor, p.high_freq_factor
    return map!(freqs, freqs) do f
        wavelen = 2π / f
        wavelen < old_context_len / high && return f
        wavelen > old_context_len / low && return f / factor
        smooth = (old_context_len / wavelen - low) / (high - low)
        return (1 - smooth) * f / factor + smooth * f
    end
end

# The tables of `NNlib.apply_rotary_embedding` for the positions 0:n-1, transformers'
# `position_embeddings`, computed in Float64 (transformers computes them in Float32) and
# returned in the element type and on the device of `x`. Its default pairing of the features
# k and k + head_dim ÷ 2 is transformers' `rotate_half`.
function rotary_tables(m::RotaryEmbedding, x, n)
    return ChainRulesCore.ignore_derivatives() do
        angles = inv_freq(m.rope_parameters, m.head_dim, x) .* (0:(n - 1))'
        T = eltype(x)
        return T.(cos.(angles)), T.(sin.(angles))
    end
end

# A model without positional embeddings (NoPE)
rotary_tables(::Nothing, x, n) = nothing

# The `window` of `NNlib.scaled_dot_product_attention` with `is_causal`: the query `i`
# attends to the keys `i - sliding_window < j ≤ i`, as in transformers'
# `sliding_window_overlay`. `nothing` where the window holds the whole sequence, which
# leaves cuDNN's fused kernel available.
attention_window(n, sliding_window) =
    sliding_window === nothing || sliding_window >= n ? nothing : sliding_window

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
device_indices(x, index) = copyto!(similar(x, Int, size(index)), index)
ChainRulesCore.@non_differentiable device_indices(::Any...)

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

# `act(gate) .* up`, for the halves `gate` and `up` of the first dimension of `h`, as
# `act(gate) * up` with `gate, up = h.chunk(2, dim=-1)`.
function gated(act, h)
    m = size(h, 1) ÷ 2
    halves = reshape(h, m, 2, :)
    y = act.(view(halves, :, 1, :)) .* view(halves, :, 2, :)
    return reshape(y, m, size(h)[2:end]...)
end

# The tokens of each expert, for the experts `index` (k, n) of the tokens, copied to the
# CPU: a matrix (capacity, num_experts) whose column `e` holds the tokens routed to `e`, in
# order, padded with token 1 to the largest number of tokens of an expert, and the matrix
# (k, n) of the positions in it of the tokens' experts.
function expert_slots(index, num_experts)
    index = Array(index)
    k = size(index, 1)
    counts = zeros(Int, num_experts)
    for e in index
        1 <= e <= num_experts || error("expert $e is not in 1:$num_experts")
        counts[e] += 1
    end
    capacity = maximum(counts; init = 0)
    token = ones(Int, capacity, num_experts)
    slot = similar(index, Int)
    fill!(counts, 0)
    for (i, e) in enumerate(index)
        c = counts[e] += 1
        token[c, e] = (i - 1) ÷ k + 1
        slot[i] = (e - 1) * capacity + c
    end
    return token, slot
end
ChainRulesCore.@non_differentiable expert_slots(::Any...)

"""
    moe_experts(gate_up, down, act, x, top_k_index, top_k_weights)

The experts of a sparse mixture of experts, as transformers' `MixtralExperts`: each token,
a column of `x` (hidden, n), goes through its experts `top_k_index` (k, n), whose outputs
`down * (act(gate) .* up)` are summed with the weights `top_k_weights` (k, n). The stacked
weights of the experts are torch's `gate_up_proj` (num_experts, 2 * intermediate, hidden)
and `down_proj` (num_experts, hidden, intermediate), so `gate_up` has size
(hidden, 2 * intermediate, num_experts) and `down` (intermediate, hidden, num_experts).

The tokens are grouped by expert and padded to the largest group, so that each projection is
one `batched_mul`, and each token gathers back the outputs of its experts. Unlike a scatter,
this needs no atomic additions, which are nondeterministic, and slow in `BFloat16` on GPUs
before sm_90.
"""
function moe_experts(gate_up, down, act, x, top_k_index, top_k_weights)
    hidden, n = size(x)
    k = size(top_k_index, 1)
    token, slot = expert_slots(top_k_index, size(gate_up, 3))
    xs = NNlib.gather(x, device_indices(x, token))
    h = gated(act, batched_mul(batched_transpose(gate_up), xs))
    y = reshape(batched_mul(batched_transpose(down), h), hidden, :)
    ys = NNlib.gather(y, device_indices(x, slot)) .* reshape(top_k_weights, 1, k, n)
    return dropdims(sum(ys; dims = 2); dims = 2)
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
