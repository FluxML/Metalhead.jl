# Ported from sentence_transformers.sentence_transformer.modules.pooling
# (sentence-transformers 6.2.0, Apache-2.0, UKP Lab and the Hugging Face Team), for padded
# inputs. Validated against test/fixtures/st_pooling.

const POOLING_MODES = ("cls", "max", "mean", "mean_sqrt_len_tokens", "weightedmean",
    "lasttoken")

"""
    Pooling(embedding_dimension; pooling_mode = "mean", include_prompt = true)

sentence-transformers' `Pooling` module: the `sentence_embedding` of each text, of size
`(embedding_dimension * length(pooling_mode), batch)`, from the `token_embeddings` of size
`(embedding_dimension, seq_len, batch)` and the `attention_mask` of a NamedTuple of
features, by each mode of `pooling_mode` (a string or a tuple of them, whose embeddings are
concatenated): `"cls"`, the first token; `"lasttoken"`, the last one; `"mean"`, `"max"`, and
`"mean_sqrt_len_tokens"`, the sum divided by the square root of the number of tokens;
`"weightedmean"`, the mean weighted by the position of each token (from 1). The padding,
on the right or on the left, is left out, and so are the first `prompt_length` tokens of
the features with `include_prompt = false`.
"""
@concrete struct Pooling
    embedding_dimension
    pooling_mode
    include_prompt
end
Flux.@layer Pooling

function Pooling(embedding_dimension::Integer; pooling_mode = "mean",
        include_prompt::Bool = true)
    modes = pooling_mode isa AbstractString ? (String(pooling_mode),) :
            Tuple(String(m) for m in pooling_mode)
    for mode in modes
        mode in POOLING_MODES ||
            throw(ArgumentError("`pooling_mode = $(repr(mode))` is not a pooling mode; the modes are $(join(repr.(POOLING_MODES), ", "))"))
    end
    return Pooling(Int(embedding_dimension), modes, include_prompt)
end

function (m::Pooling)(features::NamedTuple)
    x = features.token_embeddings
    _, n, b = size(x)
    mask = get(features, :attention_mask, nothing)
    mask = mask !== nothing && size(mask, 1) == n ? mask : ones(Int, n, b)
    prompt = m.include_prompt ? nothing : get(features, :prompt_length, nothing)
    w = pooling_weights(Array(mask), prompt)
    pooled = [pool(Val(Symbol(mode)), x, w) for mode in m.pooling_mode]
    sentence_embedding = length(pooled) == 1 ? only(pooled) : reduce(vcat, pooled)
    return merge(features, (; sentence_embedding))
end

# The positions that each mode pools, computed on the CPU from the attention mask
# `original`, of size (seq_len, batch): `mask`, the tokens pooled (without the prompt of
# `prompt_length` tokens after the left padding), as 0 and 1; `first` and `last`, the first
# and last of them (the first position where there are none); and `weights`, the positions
# from the first token of the text, 1-based, of the tokens pooled.
function pooling_weights(original::AbstractMatrix, prompt_length)
    n, b = size(original)
    pads = [something(findfirst(!iszero, view(original, :, j)), 1) - 1 for j in 1:b]
    skip = something(prompt_length, 0)
    mask = [original[i, j] != 0 && (skip == 0 || i - 1 >= pads[j] + skip)
            for i in 1:n,
    j in 1:b]
    first = [something(findfirst(view(mask, :, j)), 1) for j in 1:b]
    last = [something(findlast(view(mask, :, j)), 1) for j in 1:b]
    weights = [mask[i, j] * (i - pads[j]) for i in 1:n, j in 1:b]
    return (; mask, first, last, weights)
end
ChainRulesCore.@non_differentiable pooling_weights(::Any...)

# The accumulation type of sums of `x`, at least `Float32`, as sentence-transformers' for
# 16-bit embeddings.
accumulation_type(x) = promote_type(eltype(x), Float32)

# `a`, an array on the CPU, as an array of eltype `T` on the device of `x`.
device_array(x, T, a) = copyto!(similar(x, T, size(a)), a)
ChainRulesCore.@non_differentiable device_array(::Any...)

# The columns `index` of the batch elements of `x`, of size (dim, seq_len, batch).
function take_tokens(x, index)
    d, n, _ = size(x)
    columns = [(j - 1) * n + i for (j, i) in enumerate(index)]
    return NNlib.gather(reshape(x, d, :), device_array(x, Int, columns))
end

pool(::Val{:cls}, x, w) = take_tokens(x, w.first)

function pool(::Val{:lasttoken}, x, w)
    valid = [w.mask[i, j] for (j, i) in enumerate(w.last)]
    return take_tokens(x, w.last) .* device_array(x, eltype(x), reshape(valid, 1, :))
end

function pool(::Val{:max}, x, w)
    mask = device_array(x, Bool, reshape(w.mask, 1, size(w.mask)...))
    return dropdims(maximum(ifelse.(mask, x, typemin(eltype(x))); dims = 2); dims = 2)
end

# The sums of the tokens of `x` weighted by `weights`, divided by `denominator` of the sums
# of the weights, in the accumulation type, converted to the eltype of `x`.
function weighted_sum(x, weights, denominator)
    A = accumulation_type(x)
    w = device_array(x, A, reshape(weights, 1, size(weights)...))
    total = dropdims(sum(x .* w; dims = 2); dims = 2)
    d = device_array(x, A, reshape(denominator.(max.(sum(weights; dims = 1), 1e-9)), 1, :))
    y = total ./ d
    return eltype(y) == eltype(x) ? y : eltype(x).(y)
end

pool(::Val{:mean}, x, w) = weighted_sum(x, w.mask, identity)
pool(::Val{:mean_sqrt_len_tokens}, x, w) = weighted_sum(x, w.mask, sqrt)
pool(::Val{:weightedmean}, x, w) = weighted_sum(x, w.weights, identity)

const POOLING_LEGACY_KEYS = ("pooling_mode_cls_token" => "cls",
    "pooling_mode_max_tokens" => "max", "pooling_mode_mean_tokens" => "mean",
    "pooling_mode_mean_sqrt_len_tokens" => "mean_sqrt_len_tokens",
    "pooling_mode_weightedmean_tokens" => "weightedmean",
    "pooling_mode_lasttoken" => "lasttoken")

# The configuration of a pooling with the modes of the booleans of sentence-transformers < 5
# as its `pooling_mode`, as sentence-transformers' `_convert_legacy_pooling_kwargs`: the
# modes set, in this order, or "mean" if none is.
function convert_legacy_pooling(dict)
    legacy = filter(p -> any(==(first(p)), first.(POOLING_LEGACY_KEYS)), dict)
    isempty(legacy) && return dict
    converted = filter(p -> !haskey(legacy, first(p)), dict)
    if !haskey(dict, "pooling_mode")
        modes = [mode
                 for (key, mode) in POOLING_LEGACY_KEYS if get(legacy, key, false) == true]
        isempty(modes) && push!(modes, "mean")
        converted["pooling_mode"] = length(modes) == 1 ? only(modes) : modes
    end
    return converted
end

function load_pooling(source; pretrained, eltype)
    file = "config.json"
    dict = read_file(source, file)
    dict === nothing &&
        error("the Pooling module of $(source_name(source)) has no config.json")
    c = read_settings(Pooling, convert_legacy_pooling(dict), file;
        defaults = (; embedding_dimension = nothing, pooling_mode = "mean",
            include_prompt = true),
        renamed = (; word_embedding_dimension = :embedding_dimension))
    c.embedding_dimension isa Integer ||
        error("the Pooling module of $(source_name(source)) has no `embedding_dimension`")
    for mode in (c.pooling_mode isa AbstractString ? (c.pooling_mode,) : c.pooling_mode)
        mode in POOLING_MODES || unsupported("pooling_mode", mode, file, POOLING_MODES)
    end
    return Pooling(c.embedding_dimension; c.pooling_mode, c.include_prompt)
end
MODULE_LOADERS["Pooling"] = load_pooling
