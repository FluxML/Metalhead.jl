# Utility function for classifier head of vision transformer-like models
seconddimmean(x) = dropdims(mean(x; dims = 2); dims = 2)

# utility function for making sure that all layers have a channel size divisible by 8
# used by MobileNet variants
function _round_channels(channels, divisor, min_value = divisor)
    new_channels = max(min_value, floor(Int, channels + divisor / 2) ÷ divisor * divisor)
    # Make sure that round down does not go down by more than 10%
    return new_channels < 0.9 * channels ? new_channels + divisor : new_channels
end

"""
    addact(activation = relu, xs...)

Convenience function for applying an activation function to the output after
summing up the input arrays. Useful as the `connection` argument for the block
function in [`resnet`](#).

See also [`reluadd`](#).
"""
addact(activation = relu, xs...) = activation(sum(xs))

"""
    actadd(activation = relu, xs...)

Convenience function for adding input arrays after applying an activation
function to them. Useful as the `connection` argument for the block function in
[`resnet`](#).

See also [`addrelu`](#).
"""
actadd(activation = relu, xs...) = sum(activation.(x) for x in xs)

"""
    cat_channels(x, y, zs...)

Concatenate `x` and `y` (and any `z`s) along the channel dimension (third dimension).
Equivalent to `cat(x, y, zs...; dims=3)`.
Convenient reduction operator for use with `Parallel`.
"""
cat_channels(xs::AbstractArray...) = cat(xs...; dims = Val(3))
cat_channels(x::AbstractArray, y::Tuple) = cat_channels(x, y...)
cat_channels(x::Tuple, y::AbstractArray...) = cat_channels(x..., y...)
cat_channels(x::Tuple) = cat_channels(x...)

"""
    swapdims(perm)

Convenience function for permuting the dimensions of an array.
`perm` is a vector or tuple specifying a permutation of the input dimensions.
Equivalent to `permutedims(x, perm)`.
"""
swapdims(perm) = Base.Fix2(permutedims, perm)

# Utility function for pretty printing large models
function _maybe_big_show(io, model)
    if isdefined(Flux, :_big_show)
        if isnothing(get(io, :typeinfo, nothing)) # e.g. top level in REPL
            Flux._big_show(io, model)
        else
            show(io, model)
        end
    else
        show(io, model)
    end
end

"""
    linear_scheduler(drop_rate = 0.0; start_value = 0.0, depth)
    linear_scheduler(drop_rate::Nothing; depth::Integer)

Returns the dropout rates for a given depth using the linear scaling rule. If the
`drop_rate` is `nothing`, it returns a `Vector` of length `depth` with all values
equal to `nothing`.
"""
function linear_scheduler(drop_rate = 0.0; depth::Integer, start_value = 0.0)
    return LinRange(start_value, drop_rate, depth)
end
linear_scheduler(drop_rate::Nothing; depth::Integer) = fill(drop_rate, depth)

# Utility function for depth and configuration checks in models
function _checkconfig(config, configs)
    @assert config in configs "Invalid configuration. Must be one of $(sort(collect(configs)))."
end
