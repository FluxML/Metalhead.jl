# Utility function for classifier head of vision transformer-like models
_seconddimmean(x) = dropdims(mean(x, dims = 2); dims = 2)

# utility function for making sure that all layers have a channel size divisible by 8
# used by MobileNet variants
function _round_channels(channels, divisor, min_value = divisor)
  new_channels = max(min_value, floor(Int, channels + divisor / 2) ÷ divisor * divisor)
  # Make sure that round down does not go down by more than 10%
  return (new_channels < 0.9 * channels) ? new_channels + divisor : new_channels
end

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
