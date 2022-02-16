# Utility function for classifier head of vision transformer-like models
_seconddimmean(x) = dropdims(mean(x, dims = 2); dims = 2)

"""
    addrelu(x, y)

Convenience function for `(x, y) -> @. relu(x + y)`.
Useful as the `connection` argument for [`resnet`](#).
See also [`reluadd`](#).
"""
addrelu(x, y) = @. relu(x + y)

"""
    reluadd(x, y)

Convenience function for `(x, y) -> @. relu(x) + relu(y)`.
Useful as the `connection` argument for [`resnet`](#).
See also [`addrelu`](#).
"""
reluadd(x, y) = @. relu(x) + relu(y)

"""
    cat_channels(x, y)

Concatenate `x` and `y` along the channel dimension (third dimension).
Equivalent to `cat(x, y; dims=3)`.
Convenient binary reduction operator for use with `Parallel`.
"""
cat_channels(x, y) = cat(x, y; dims = 3)
