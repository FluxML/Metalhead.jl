# The key-value cache of the causal language models whose layers are all full attention.
# The layers write and read it through `LayerCache` and `attend` (layers.jl).

# The attention layers that take a `LayerCache`
const CachedAttention = Union{LlamaAttention, Qwen3Attention, GraniteAttention}

cacheable(::CausalLM) = false
function cacheable(m::Union{LlamaForCausalLM, GraniteForCausalLM})
    return all(layer -> layer.self_attn isa CachedAttention, m.model.layers)
end

function check_cacheable(m::CausalLM)
    cacheable(m) ||
        error("a key-value cache is not supported yet for a $(nameof(typeof(m))): only the models whose layers are all full attention have one (Llama, Qwen2, Qwen3, Qwen3 MoE and Granite), not those with sliding-window attention (Gemma 3) or a gated delta rule (Qwen3.5)")
    return nothing
end

"""
    KVCache(model, batch_size, max_len)

The key-value cache of a causal language model for `batch_size` sequences of up to `max_len`
tokens each. With it, each step of a generation loop runs the model on the new tokens only,
rather than on the whole sequence so far. It holds the keys and values of the tokens seen,
for each layer, in buffers allocated at their full length, with the type and on the device
of the model's weights, and `lengths`, a `Vector{Int}` on the host with the number of tokens
of each sequence. The keys are stored rotated.

Llama, Qwen2, Qwen3, Qwen3 MoE and the Granite models (their attention layers are all full
attention) have a cache; Gemma 3 (sliding-window attention) and Qwen3.5 (a gated delta
rule) raise a "not supported yet" error.

`model(input_ids; cache)` runs the model on the token indices `input_ids` of size
`(n, batch_size)`, which follow the tokens in the cache, stores their keys and values and
returns `(logits, cache)`, the cache being updated in place;
`forward_features(model, input_ids; cache)` returns `(features, cache)` the same way. A
vector of token indices is a batch of one. The rotary positions of the new tokens continue
those of their sequence.

- The first call, with an empty cache, is the **prefill** of the prompts: a batch of
  sequences of different lengths is padded on the right, and its `attention_mask` (see
  [`llama_for_causal_lm`](@ref Metalhead.Transformers.llama_for_causal_lm)) sets the
  `lengths`. It computes what the model computes without a cache. The logits of the next
  token of a sequence are at its last token, `logits[:, lengths[b], b]`, which is not the
  last position of a shorter one: [`last_tokens`](@ref Metalhead.Transformers.last_tokens)
  picks them.
- The next calls are **steps** of `n ≥ 1` new tokens (the same number for all the
  sequences, without `attention_mask`), written after the tokens of each sequence, whatever
  the lengths of the others. A step of a few tokens, such as the chunk of a long prompt,
  attends to the cache and, causally, to the tokens of the chunk. Their logits are those of
  the same tokens in a forward pass of the whole sequence, up to rounding.

The sequences of a batch of the same length write their keys with one copy; otherwise, each
sequence writes its own. `max_len` bounds the first call too, padding included. The cache is
for inference: gradients do not flow through it. `empty!(cache)` makes it ready for new
sequences.

# Examples

A greedy loop over a batch of two texts of different lengths (`tok` is a
[`TextProcessor`](@ref Metalhead.TextProcessor)):

```julia
using Metalhead.Transformers: KVCache, last_tokens

inputs = tok(["The capital of France is", "Hello"])           # padded on the right
cache = KVCache(model, size(inputs.input_ids, 2), 64)
logits, _ = model(inputs.input_ids; inputs.attention_mask, cache)    # prefill
next = argmax.(eachcol(last_tokens(logits, cache.lengths)))     # one token per sequence
for _ in 1:8
    logits, _ = model(reshape(next, 1, :); cache)               # a step of one token
    next = argmax.(eachcol(dropdims(logits; dims = 2)))
end
```
"""
struct KVCache{A <: AbstractArray, P}
    keys::Vector{A}
    values::Vector{A}
    lengths::Vector{Int}
    position_embeddings::P
end

function KVCache(model::CausalLM, batch_size::Integer, max_len::Integer)
    check_cacheable(model)
    batch_size >= 1 && max_len >= 1 || throw(ArgumentError(
        "`batch_size` ($batch_size) and `max_len` ($max_len) must be positive"))
    decoder = model.model
    weight = decoder.embed_tokens.weight
    # zeros: the masked keys of a buffer must not be NaN, which the weights of 0 would keep
    function buffer(attention)
        kv_heads = size(attention.k_proj.weight, 1) ÷ attention.head_dim
        return fill!(similar(weight, attention.head_dim, kv_heads, max_len, batch_size), 0)
    end
    keys = [buffer(layer.self_attn) for layer in decoder.layers]
    values = [buffer(layer.self_attn) for layer in decoder.layers]
    return KVCache(keys, values, zeros(Int, batch_size),
        rotary_tables(decoder.rotary_emb, weight, max_len))
end

function Base.show(io::IO, c::KVCache)
    return print(io, "KVCache(", length(c.keys), " layers, ", length(c.lengths),
        " sequences of up to ", size(first(c.keys), 3), " tokens, lengths = ", c.lengths, ")")
end

function Base.empty!(c::KVCache)
    fill!(c.lengths, 0)
    return c
end

# `m` on the new tokens `input_ids`, after the tokens of `cache`; updates the cache.
function cached_forward(m::DecoderModel, input_ids, attention_mask, cache::KVCache)
    n, batch_size = size(input_ids)
    length(cache.keys) == length(m.layers) ||
        error("the cache has $(length(cache.keys)) layers, the model $(length(m.layers))")
    batch_size == length(cache.lengths) ||
        error("the cache is for $(length(cache.lengths)) sequences, the token indices have $batch_size")
    start = copy(cache.lengths)
    kv_lengths = kv_lengths_of(attention_mask, input_ids)
    kv_lengths === nothing || all(iszero, start) ||
        error("`attention_mask` only goes with the first call of an empty cache: later calls add the same number of tokens to every sequence")
    max_len = size(first(cache.keys), 3)
    maximum(start) + n <= max_len ||
        error("the cache holds $max_len tokens per sequence: $(maximum(start)) cached and $n new ones don't fit")
    stop = start .+ something(kv_lengths, n)
    x = input_embeddings(m, input_ids)
    offset = allequal(start) ? first(start) : device_indices(x, start)
    for (i, layer) in enumerate(m.layers)
        layer_cache = LayerCache(cache.keys[i], cache.values[i], start, stop, offset)
        x = layer(x, cache.position_embeddings, kv_lengths, layer_cache)
    end
    cache.lengths .= stop
    return m.norm(x), cache
end

"""
    last_tokens(x, lengths)

The columns of `x`, of size `(features, seq_len, batch_size)`, at the position `lengths[b]`
of each sequence `b`: an array of size `(features, batch_size)`. For the logits of a
[`KVCache`](@ref Metalhead.Transformers.KVCache) prefill, with `cache.lengths` as `lengths`,
these are the logits of the next token of each sequence of a batch padded on the right,
whose last position is padding for the shorter ones.
"""
function last_tokens(x::AbstractArray{<:Any, 3}, lengths::AbstractVector{<:Integer})
    length(lengths) == size(x, 3) ||
        throw(DimensionMismatch("$(length(lengths)) lengths for $(size(x, 3)) sequences"))
    all(l -> 1 <= l <= size(x, 2), lengths) ||
        error("the lengths $lengths are not within the $(size(x, 2)) positions")
    return reduce(hcat, [x[:, l, b] for (b, l) in enumerate(lengths)])
end
