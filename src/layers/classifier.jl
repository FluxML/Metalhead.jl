"""
    create_classifier(inplanes, nclasses; pool_type = :mean, use_conv = false)

Creates a classifier head to be used for models. Uses `SelectAdaptivePool` for the pooling layer.

# Arguments

  - `inplanes`: number of input feature maps
  - `nclasses`: number of output classes
  - `pool_type`: the type of adaptive pooling to use. One of `:mean`, `:max`, `:meanmax`,
    `:catmeanmax` or `:identity`.
  - `use_conv`: whether to use a 1x1 convolutional layer instead of a `Dense` layer.
"""
function create_classifier(inplanes, nclasses; pool_type = :mean, use_conv = false)
    flatten_in_pool = !use_conv  # flatten when we use a Dense layer after pooling
    if pool_type == :identity
        @assert use_conv
        "Pooling can only be disabled if classifier is also removed or a convolution-based classifier is used"
        flatten_in_pool = false  # disable flattening if pooling is pass-through (no pooling)
    end
    global_pool = SelectAdaptivePool(; pool_type, flatten = flatten_in_pool)
    fc = use_conv ? Conv((1, 1), inplanes => nclasses; bias = true) :
         Dense(inplanes => nclasses; bias = true)
    return global_pool, fc
end
