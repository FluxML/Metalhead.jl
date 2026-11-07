# transformers' `generate` for the causal language models, by greedy decoding or multinomial
# sampling: the settings of a checkpoint (`GenerationConfig`, generation/configuration_utils.py),
# the logits processors that they enable (generation/logits_process.py, in the order of
# `GenerationMixin._get_logits_processor`) and the loop of `GenerationMixin._sample`
# (generation/utils.py).

"""
    GenerationConfig(; kwargs...)
    GenerationConfig(config::GenerationConfig; kwargs...)

The settings of [`generate`](@ref), with the names and defaults of
transformers' `GenerationConfig`, for greedy decoding and multinomial sampling; the second
form changes the settings `kwargs` of `config`.

- `max_new_tokens`: the number of tokens to generate at most; or `max_length`, the length of
  the prompt (padded) and the new tokens; neither: 20 new tokens.
- `min_new_tokens`, or `min_length`: until then, the end-of-text tokens are excluded.
- `do_sample = false`: the most likely token; `true` samples the next token from the
  logits divided by `temperature = 1.0`, among the `top_k = 50` most likely tokens (`0`: all
  of them), then the most likely of cumulative probability `top_p = 1.0`, then those of
  probability at least `min_p` times that of the most likely (`nothing`: all).
- `repetition_penalty = 1.0`: divides the positive logits of the tokens that a sequence
  has, and multiplies the negative ones.
- `eos_token_id`: the end-of-text tokens, which end a sequence (an integer, or several);
  `pad_token_id`, the token of a sequence after its end (the first end-of-text token by
  default); `bos_token_id`. These are ids of a Hugging Face tokenizer, from 0, as in the
  files: token indices are one more.

The `generation_config` of a language model that `create_model` loads is that of its
checkpoint, as transformers' `model.generation_config`: its `generation_config.json`, or the
token ids of its `config.json` (of its `text_config` too) without one. A
`generation_config.json` that asks for anything else, such as beam search
(`num_beams > 1`), `no_repeat_ngram_size` or `suppress_tokens`, is an error.
"""
struct GenerationConfig
    max_length::Union{Nothing, Int}
    max_new_tokens::Union{Nothing, Int}
    min_length::Int
    min_new_tokens::Union{Nothing, Int}
    do_sample::Bool
    temperature::Float64
    top_k::Int
    top_p::Float64
    min_p::Union{Nothing, Float64}
    repetition_penalty::Float64
    bos_token_id::Union{Nothing, Int}
    eos_token_id::Tuple{Vararg{Int}}
    pad_token_id::Union{Nothing, Int}
end
Functors.@leaf GenerationConfig

# transformers' defaults (`GenerationConfig._get_default_generation_params`)
const GENERATION_DEFAULTS = (; max_length = nothing, max_new_tokens = nothing,
    min_length = 0, min_new_tokens = nothing, do_sample = false, temperature = 1.0,
    top_k = 50, top_p = 1.0, min_p = nothing, repetition_penalty = 1.0,
    bos_token_id = nothing, eos_token_id = (), pad_token_id = nothing)

token_ids(::Nothing) = ()
token_ids(id::Integer) = (Int(id),)
token_ids(ids) = Tuple{Vararg{Int}}(Int.(Tuple(ids)))

function GenerationConfig(c::GenerationConfig; kwargs...)
    for k in keys(kwargs)
        haskey(GENERATION_DEFAULTS, k) || throw(ArgumentError(not_a_setting(k)))
    end
    s = merge(NamedTuple{fieldnames(GenerationConfig)}(getfield.(Ref(c),
            fieldnames(GenerationConfig))), values(kwargs))
    return GenerationConfig(s.max_length, s.max_new_tokens, s.min_length, s.min_new_tokens,
        s.do_sample, s.temperature, s.top_k, s.top_p, s.min_p, s.repetition_penalty,
        s.bos_token_id, token_ids(s.eos_token_id), s.pad_token_id)
end
GenerationConfig(; kwargs...) = GenerationConfig(DEFAULT_GENERATION; kwargs...)
const DEFAULT_GENERATION = GenerationConfig(values(GENERATION_DEFAULTS)...)

function Base.show(io::IO, c::GenerationConfig)
    set = [k for k in fieldnames(GenerationConfig)
           if getfield(c, k) != getfield(DEFAULT_GENERATION, k)]
    return print(io, "GenerationConfig(",
        join(("$k = $(repr(getfield(c, k)))" for k in set), ", "), ")")
end

function not_a_setting(k)
    supported = join(keys(GENERATION_DEFAULTS), ", ")
    haskey(UNSUPPORTED_GENERATION, k) &&
        return "`$k` is not supported by `generate`, which decodes greedily or samples; its settings are $supported"
    return "`$k` is not a setting of `generate`: $supported"
end

# The settings of transformers' `GenerationConfig` that `generate` doesn't support, with the
# values at which they change nothing (and `nothing`).
const UNSUPPORTED_GENERATION = (; num_beams = 1, num_beam_groups = 1,
    diversity_penalty = 0.0, penalty_alpha = nothing, dola_layers = nothing,
    typical_p = 1.0, epsilon_cutoff = 0.0, eta_cutoff = 0.0, top_h = nothing,
    encoder_repetition_penalty = 1.0, no_repeat_ngram_size = 0,
    encoder_no_repeat_ngram_size = 0, bad_words_ids = nothing, force_words_ids = nothing,
    constraints = nothing, sequence_bias = nothing, suppress_tokens = nothing,
    begin_suppress_tokens = nothing, forced_bos_token_id = nothing,
    forced_eos_token_id = nothing, remove_invalid_values = false,
    exponential_decay_length_penalty = nothing, guidance_scale = 1.0,
    watermarking_config = nothing, num_return_sequences = 1, max_time = nothing,
    stop_strings = nothing, token_healing = false, use_mtp = false,
    prompt_lookup_num_tokens = nothing, max_matching_ngram_size = nothing,
    assistant_early_exit = nothing)

# The settings that change neither the tokens of greedy decoding nor the distribution of
# sampling: what `generate` returns and caches, the logits normalized after the processors,
# those of beam search, encoder-decoder models and assistant models.
const IGNORED_GENERATION = (:transformers_version, :_from_model_config, :use_cache,
    :cache_implementation, :cache_config, :max_cache_len, :prefill_chunk_size,
    :compile_config, :disable_compile, :continuous_batching_config, :output_attentions,
    :output_hidden_states, :output_scores, :output_logits, :return_dict_in_generate,
    :renormalize_logits, :early_stopping, :length_penalty, :low_memory,
    :decoder_start_token_id, :is_assistant, :num_assistant_tokens,
    :num_assistant_tokens_schedule, :assistant_confidence_threshold, :assistant_lookbehind,
    :target_lookbehind, :assistant_ensemble_weight, :speculation_type)

changes_nothing(v, default) = v === nothing || v == default || v isa AbstractVector && isempty(v)

# The settings of `generate` in the parsed `dict` of `file`, whose `null`s are unset; with
# `others = :warn`, a key that isn't a setting of transformers' `GenerationConfig` warns.
function generation_settings(dict, file; others = :warn)
    settings = Dict{Symbol, Any}()
    for (k, v) in dict
        key = Symbol(k)
        if haskey(GENERATION_DEFAULTS, key)
            v === nothing || (settings[key] = jsonvalue(v))
        elseif haskey(UNSUPPORTED_GENERATION, key)
            changes_nothing(v, UNSUPPORTED_GENERATION[key]) ||
                throw(ArgumentError("`$key = $(repr(v))` in $file is not supported by `generate`, which decodes greedily or samples"))
        elseif others === :warn && !(key in IGNORED_GENERATION)
            warn_unknown(GenerationConfig, key, file)
        end
    end
    return settings
end

# The generation config of a checkpoint (transformers' `from_pretrained`): its
# generation_config.json or, without one or without weights, `GenerationConfig.from_model_config`
# of its config.json, the settings of the top level, then those of the text config unset there.
function generation_config(source, config; pretrained::Bool)
    path = pretrained ? source_file_if_exists(source, "generation_config.json") : nothing
    path === nothing || return GenerationConfig(;
        generation_settings(JSON.parsefile(path), "generation_config.json")...)
    settings = generation_settings(config, "config.json"; others = :skip)
    i = findfirst(k -> get(config, k, nothing) isa AbstractDict,
        ("decoder", "generator", "text_config"))
    if i !== nothing
        text = config[("decoder", "generator", "text_config")[i]]
        settings = merge(generation_settings(text, "config.json"; others = :skip), settings)
    end
    return GenerationConfig(; settings...)
end

# `m` with the generation config `c`
function with_generation_config(m::CausalLM, c::GenerationConfig)
    T = typeof(m)
    return T.name.wrapper((f === :generation_config ? c : getfield(m, f)
                           for f in fieldnames(T))...)
end

function instantiate(format::TransformersFormat, spec, source, config; pretrained::Bool,
        kwargs...)
    model = invoke(instantiate, Tuple{AbstractConfigFormat, Any, Any, Any}, format, spec,
        source, config; pretrained, kwargs...)
    model isa CausalLM || return model
    return with_generation_config(model, generation_config(source, config; pretrained))
end

function check_generation(c::GenerationConfig)
    c.max_new_tokens === nothing || c.max_new_tokens > 0 ||
        throw(ArgumentError("`max_new_tokens` must be positive, not $(c.max_new_tokens)"))
    c.repetition_penalty > 0 ||
        throw(ArgumentError("`repetition_penalty` must be positive, not $(c.repetition_penalty)"))
    c.do_sample || return nothing
    c.temperature > 0 ||
        throw(ArgumentError("`temperature` must be positive, not $(c.temperature); for greedy decoding, set `do_sample = false`"))
    c.top_k >= 0 || throw(ArgumentError("`top_k` must be positive, or 0, not $(c.top_k)"))
    0 <= c.top_p <= 1 || throw(ArgumentError("`top_p` must be in [0, 1], not $(c.top_p)"))
    c.min_p === nothing || 0 <= c.min_p <= 1 ||
        throw(ArgumentError("`min_p` must be in [0, 1], not $(c.min_p)"))
    return nothing
end

# The number of new tokens, after a prompt of `prompt_length` tokens (transformers' `max_length`
# being the length of the prompt and the new tokens, at least one of them).
function new_tokens(c::GenerationConfig, prompt_length)
    c.max_new_tokens === nothing || return c.max_new_tokens
    c.max_length === nothing && return 20
    return max(c.max_length - prompt_length, 1)
end

# Whether the end-of-text tokens are excluded from the token of `step` (from 1) after a prompt
# of `prompt_length` tokens: transformers' `MinLengthLogitsProcessor` and
# `MinNewTokensLengthLogitsProcessor`.
function excludes_eos(c::GenerationConfig, prompt_length, step)
    isempty(c.eos_token_id) && return false
    prompt_length + step - 1 < c.min_length && return true
    return c.min_new_tokens !== nothing && step - 1 < c.min_new_tokens
end

# The logits processors (transformers' order), on the Float32 logits of the next token of
# each sequence, `scores` of size `(vocab_size, batch_size)`: the tokens that each sequence
# has (`seen`), the end-of-text tokens (`eos`), excluded while `short`.
function process_logits(c::GenerationConfig, scores, seen, eos, short::Bool)
    if c.repetition_penalty != 1
        p = Float32(c.repetition_penalty)
        scores = ifelse.(seen, ifelse.(scores .< 0, scores .* p, scores ./ p), scores)
    end
    short && (scores = ifelse.(eos, -Inf32, scores))
    c.do_sample || return scores
    c.temperature == 1 || (scores = scores ./ Float32(c.temperature))
    if c.top_k != 0
        scores = ifelse.(scores .< kth_largest(scores, min(c.top_k, size(scores, 1))),
            -Inf32, scores)
    end
    c.top_p < 1 && (scores = ifelse.(scores .< top_p_cutoffs(scores, c.top_p), -Inf32, scores))
    if c.min_p !== nothing
        probs = softmax(scores; dims = 1)
        scores = ifelse.(probs .< Float32(c.min_p) .* maximum(probs; dims = 1), -Inf32, scores)
    end
    return scores
end

kth_largest(x::Matrix, k) = reshape([partialsort(c, k; rev = true) for c in eachcol(x)], 1, :)
kth_largest(x::AbstractMatrix, k) = sort(x; dims = 1, rev = true)[k:k, :]

# The smallest logit of each column that top-p sampling keeps, from the logits of the column
# in ascending order: the most likely tokens of cumulative probability above `top_p`, at
# least one. On the CPU, of the tokens that top-k leaves.
function top_p_cutoff(sorted, top_p)
    cumulative = cumsum(softmax(sorted; dims = 1); dims = 1)
    keep = (cumulative .> Float32(1 - top_p)) .| (axes(sorted, 1) .== size(sorted, 1))
    return minimum(ifelse.(keep, sorted, Inf32); dims = 1)
end
function top_p_cutoffs(x::Matrix, top_p)
    return reduce(hcat, [top_p_cutoff(sort!(filter(>(-Inf32), c)), top_p) for c in eachcol(x)])
end
top_p_cutoffs(x::AbstractMatrix, top_p) = top_p_cutoff(sort(x; dims = 1), top_p)

greedy_tokens(scores) = map(i -> i[1], vec(Array(argmax(scores; dims = 1))))

# A token of each column from the probabilities of `softmax(scores)`, by inverting their
# cumulative distribution at a uniform number drawn on the host.
function sample_tokens(rng, scores)
    cdf = cumsum(softmax(scores; dims = 1); dims = 1)
    u = to_device_of(cdf, rand(rng, Float32, 1, size(cdf, 2)))
    return vec(Array(sum(cdf .< (1 .- u) .* cdf[end:end, :]; dims = 1))) .+ 1
end

"""
    generate(model, input_ids; attention_mask = nothing, generation_config = model.generation_config,
             rng = Random.default_rng(), kwargs...)
    generate(model, inputs::NamedTuple; kwargs...)

The tokens that the language `model` generates after each sequence of the token indices
`input_ids`, of size `(seq_len, batch_size)` (padded on the right, with its `attention_mask`),
or of a vector for a single sequence, as transformers' `model.generate`: a matrix of size
`(n, batch_size)`, or a vector, of their token indices; the output of a
[`TextProcessor`](@ref Metalhead.TextProcessor) can replace `input_ids` and `attention_mask`.
These are the new tokens only, those of transformers' `generate(...)[:, seq_len:]`:
`[input_ids; generate(model, input_ids)]` is the whole sequence. A sequence that ends with an
end-of-text token is followed by padding tokens, until the others end too, or `n` reaches
`max_new_tokens`.

It decodes greedily or samples with the settings of `generation_config`, that of the
checkpoint (see [`GenerationConfig`](@ref Metalhead.Transformers.GenerationConfig)), and the
settings `kwargs` change them: `max_new_tokens`, `do_sample`, `temperature`, `top_k`,
`top_p`, `min_p`, `repetition_penalty`, `min_new_tokens`, `eos_token_id`, ... Sampling draws
from `rng`. The other keyword arguments are the inputs of the model, such as the
`pixel_values` of Qwen3.5.

The models with a [`KVCache`](@ref Metalhead.Transformers.KVCache) run on the prompts, then
on each new token alone; the others (Gemma 3 and Qwen3.5) on the whole sequences at each
step. The logits processors and the sampler run on the device of the model; on a GPU, each
step copies only the new tokens to the host.

# Examples

```julia
using Metalhead, HuggingFaceTokenizers

model = create_model("HuggingFaceTB/SmolLM2-135M")
tok = create_processor("HuggingFaceTB/SmolLM2-135M")
new = generate(model, tok("The capital of France is"); max_new_tokens = 16)
decode(tok, new)
generate(model, tok(["Hello,", "The weather"]); do_sample = true, temperature = 0.7)
```
"""
function generate(m::CausalLM, inputs::NamedTuple; kwargs...)
    input_ids, model_inputs = token_inputs(m, inputs)
    given = (; (k => v for (k, v) in pairs(model_inputs) if v !== nothing)...)
    return generate(m, input_ids; given..., kwargs...)
end

# copies: on Julia 1.10, a reshaped vector can't grow
function generate(m::CausalLM, input_ids::AbstractVector{<:Integer}; attention_mask = nothing,
        kwargs...)
    mask = attention_mask === nothing ? nothing : reshape(Array(attention_mask), :, 1)
    return vec(generate(m, reshape(Array{Int}(input_ids), :, 1); attention_mask = mask,
        kwargs...))
end

function generate(m::CausalLM, input_ids::AbstractMatrix{<:Integer}; attention_mask = nothing,
        generation_config::GenerationConfig = m.generation_config,
        rng::AbstractRNG = Random.default_rng(), kwargs...)
    model_inputs = Base.structdiff(values(kwargs), GENERATION_DEFAULTS)
    for k in keys(model_inputs)
        k in input_keys(m) || throw(ArgumentError(not_a_setting(k)))
    end
    c = GenerationConfig(generation_config; Base.structdiff(values(kwargs), model_inputs)...)
    check_generation(c)
    ids = Array{Int}(input_ids)
    mask = attention_mask === nothing ? nothing : Array(attention_mask)
    lengths = something(kv_lengths_of(mask, ids), fill(size(ids, 1), size(ids, 2)))
    n = new_tokens(c, size(ids, 1))
    eos = collect(c.eos_token_id) .+ 1
    pad = something(c.pad_token_id, isempty(eos) ? 0 : first(eos) - 1) + 1
    decode = cacheable(m) && isempty(model_inputs) ?
             cached_decoding(m, ids, mask, size(ids, 1) + n - 1) :
             full_decoding(m, ids, lengths, pad, model_inputs)
    logits = decode(nothing)
    on_device(a) = copyto!(similar(logits, eltype(a), size(a)), a)
    vocab = axes(logits, 1)
    eos_mask = on_device(Vector{Bool}(vocab .∈ Ref(eos)))
    seen = nothing
    if c.repetition_penalty != 1
        # tokens past the logits (an embedding larger than the head) aren't penalized
        seen_host = zeros(Bool, length(vocab), size(ids, 2))
        for b in axes(ids, 2), i in 1:lengths[b]
            ids[i, b] in vocab && (seen_host[ids[i, b], b] = true)
        end
        seen = on_device(seen_host)
    end
    done = falses(size(ids, 2))
    tokens = Vector{Int}[]
    for step in 1:n
        short = excludes_eos(c, size(ids, 1), step)
        scores = process_logits(c, Float32.(logits), seen, eos_mask, short)
        next = c.do_sample ? sample_tokens(rng, scores) : greedy_tokens(scores)
        isempty(eos) || (next[done] .= pad)
        push!(tokens, next)
        isempty(eos) || (done .|= in.(next, Ref(eos)))
        (all(done) || step == n) && break
        seen === nothing || (seen .|= vocab .== reshape(on_device(next), 1, :))
        logits = decode(next)
    end
    return stack(tokens; dims = 1)
end

# The logits of the next token of each sequence after the prompts (`decode(nothing)`), then
# after each step `decode(next)`: with a key-value cache of up to `max_len` tokens, whose
# buffers start at 256 tokens after the prompts and double when full (the `max_length` of
# some checkpoints is that of their context, 131072 tokens for Llama 3.2),
function cached_decoding(m, ids, mask, max_len)
    weight = first(decoder(m)).embed_tokens.weight
    cache = KVCache(m, size(ids, 2), min(max_len, size(ids, 1) + 256))
    function decode(next)
        if next === nothing
            attention_mask = mask === nothing ? nothing : device_indices(weight, mask)
            x, _ = forward_features(m, device_indices(weight, ids); attention_mask, cache)
            return forward_head(m, last_tokens(x, cache.lengths))
        end
        capacity = size(first(cache.keys), 3)
        maximum(cache.lengths) < capacity ||
            (cache = grown(cache, m, min(max_len, 2 * capacity)))
        x, _ = forward_features(m, device_indices(weight, reshape(next, 1, :)); cache)
        return forward_head(m, dropdims(x; dims = 2))
    end
    return decode
end

# `cache` in buffers of `max_len` tokens
function grown(cache::KVCache, m, max_len)
    new = KVCache(m, length(cache.lengths), max_len)
    for (a, b) in zip([new.keys; new.values], [cache.keys; cache.values])
        view(a, :, :, axes(b, 3), :) .= b
    end
    new.lengths .= cache.lengths
    return new
end

# or on the whole sequences, padded on the right with the token `pad`.
function full_decoding(m, ids, lengths, pad, model_inputs)
    weight = first(decoder(m)).embed_tokens.weight
    sequences = [ids[1:l, b] for (b, l) in enumerate(lengths)]
    function decode(next)
        next === nothing || (sequences = [[s; t] for (s, t) in zip(sequences, next)])
        lengths = length.(sequences)
        padded = [i <= l ? s[i] : pad for i in 1:maximum(lengths), (s, l) in zip(sequences, lengths)]
        attention_mask = allequal(lengths) ? nothing :
                         device_indices(weight, [i <= l for i in axes(padded, 1), l in lengths])
        x = forward_features(m, device_indices(weight, padded); attention_mask,
            model_inputs...)
        return forward_head(m, last_tokens(x, lengths))
    end
    return decode
end
