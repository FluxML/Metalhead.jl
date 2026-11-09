# Ported from transformers.models.upernet.modeling_upernet (transformers 5.21, Apache-2.0):
# the decode head of UperNet (Xiao et al., 2018), a pyramid pooling module on the coarsest
# feature map and a feature pyramid network over all of them, and its auxiliary FCN head.
# Feature maps are (W, H, channels, batch), fine to coarse, as a backbone gives them.

"""
    UperNetConvModule(in_channels => out_channels, kernel_size; padding = 0,
                      bias = false, dilation = 1)

transformers' `UperNetConvModule`: a convolution, a `BatchNorm` and a ReLU.
"""
@concrete struct UperNetConvModule
    conv
    batch_norm
end
Flux.@layer UperNetConvModule

function UperNetConvModule((in_channels, out_channels)::Pair{<:Integer, <:Integer},
        kernel_size::Integer; padding::Integer = 0, bias::Bool = false,
        dilation::Integer = 1)
    conv = init_layer(Conv, (kernel_size, kernel_size), in_channels => out_channels;
        pad = padding, dilation, bias)
    return UperNetConvModule(conv, init_layer(BatchNorm, out_channels, relu))
end

function (m::UperNetConvModule)(x)
    y = m.conv(x)
    training = isnothing(m.batch_norm.active) ? NNlib.within_gradient(y) : m.batch_norm.active
    if training === true && prod(size(y)) ÷ size(y, ndims(y) - 1) == 1
        throw(ArgumentError("UperNet BatchNorm expected more than 1 value per channel when training, got input of size $(size(y))."))
    end
    return m.batch_norm(y)
end

# Keep the local name used by the tests while sharing Timm's implementation.
adaptive_avg_pool(x::AbstractArray{<:Any, 4}, scale::Integer) =
    adaptive_avg_pool2d(x, (scale, scale))

# torch's `nn.AdaptiveAvgPool2d(output_size)`, for the blocks of the pyramid pooling.
struct AdaptiveAvgPool2d
    output_size::Int
end
(p::AdaptiveAvgPool2d)(x) = adaptive_avg_pool2d(x, (p.output_size, p.output_size))

"""
    UperNetPyramidPoolingBlock(pool_scale, in_channels => channels)

transformers' `UperNetPyramidPoolingBlock`: an adaptive average pooling to
`pool_scale × pool_scale` and an [`UperNetConvModule`](@ref
Metalhead.Transformers.UperNetConvModule) (1 × 1), which torch registers as the children
"0" and "1" of the block, the names of its fields.
"""
struct UperNetPyramidPoolingBlock{P, C}
    var"0"::P
    var"1"::C
end
Flux.@layer UperNetPyramidPoolingBlock

function UperNetPyramidPoolingBlock(pool_scale::Integer, channels::Pair)
    return UperNetPyramidPoolingBlock(AdaptiveAvgPool2d(pool_scale),
        UperNetConvModule(channels, 1))
end

(m::UperNetPyramidPoolingBlock)(x) = getfield(m, Symbol(1))(getfield(m, Symbol(0))(x))

# transformers' `UperNetPyramidPoolingModule`, the pyramid pooling module of PSPNet, holds
# only its blocks, registered as its children "0", "1", ...: here a vector of
# `UperNetPyramidPoolingBlock`s, one for each pool scale, and this function its forward
# pass, the outputs of the blocks resized to the map `x` (bilinear,
# `align_corners = False`).
function pyramid_pooling(psp_modules, x)
    return [interpolate_bilinear(block(x), size(x)[1:2]) for block in psp_modules]
end

# The concatenation of maps of the same size along their channels, in order, as
# torch.cat(maps, dim = 1); stack, as splatting a vector into `cat` isn't type-stable.
function cat_channels(maps)
    w, h, _, b = size(first(maps))
    all(m -> size(m, 3) == size(first(maps), 3), maps) && return reshape(
        stack(maps; dims = 4), w, h, :, b)
    return cat(maps...; dims = 3)
end

"""
    UperNetHead(in_channels; hidden_size = 512, pool_scales = (1, 2, 3, 6), num_labels)

transformers' `UperNetHead`, the decode head of UperNet: a pyramid pooling module
(`psp_modules`, `bottleneck`) on the coarsest of the feature maps of a backbone, with
`in_channels[i]` channels, fine to coarse; a feature pyramid network (`lateral_convs`,
`fpn_convs`) over the others, which adds each level to the coarser one above it; the
levels, resized to the finest and concatenated, fused by `fpn_bottleneck`; and the
`classifier`, a 1 × 1 convolution to `num_labels`. Its keyword arguments are the fields of
`UperNetConfig`.

Called on the feature maps (W, H, channels, batch), it returns the logits
(W, H, num_labels, batch) at the size of the finest, as transformers does before resizing
them to the images. Maps of any size work: each is resized to the next finer one.
"""
@concrete struct UperNetHead
    classifier
    psp_modules
    bottleneck
    lateral_convs
    fpn_convs
    fpn_bottleneck
end
Flux.@layer UperNetHead

function UperNetHead(in_channels; hidden_size::Integer = 512,
        pool_scales = (1, 2, 3, 6), num_labels::Integer)
    in_channels = collect(Int, in_channels)
    check_head_args(in_channels, hidden_size, num_labels)
    all(>(0), pool_scales) ||
        throw(ArgumentError("`pool_scales` must be positive, got $(repr(pool_scales))"))
    top = last(in_channels)
    return UperNetHead(init_layer(Conv, (1, 1), hidden_size => num_labels),
        [UperNetPyramidPoolingBlock(s, top => hidden_size) for s in pool_scales],
        UperNetConvModule(top + length(pool_scales) * hidden_size => hidden_size, 3;
            padding = 1),
        [UperNetConvModule(c => hidden_size, 1) for c in in_channels[1:(end - 1)]],
        [UperNetConvModule(hidden_size => hidden_size, 3; padding = 1)
         for _ in in_channels[1:(end - 1)]],
        UperNetConvModule(length(in_channels) * hidden_size => hidden_size, 3;
            padding = 1))
end

function check_head_args(in_channels, channels, num_labels)
    isempty(in_channels) && throw(ArgumentError("`in_channels` must not be empty"))
    all(>(0), in_channels) && channels > 0 && num_labels > 0 ||
        throw(ArgumentError("channel counts and `num_labels` must be positive"))
    return nothing
end

function check_features(features, n)
    length(features) == n ||
        throw(ArgumentError("the head takes $n feature maps, got $(length(features))"))
    return nothing
end
ChainRulesCore.@non_differentiable check_features(::Any...)

function psp_forward(m::UperNetHead, x)
    return m.bottleneck(cat_channels([[x]; pyramid_pooling(m.psp_modules, x)]))
end

# The top-down pathway: each lateral plus the coarser one above it, resized to it, from
# the coarsest down. Without mutation, for Zygote.
function top_down(laterals)
    merged = [last(laterals)]
    for i in (length(laterals) - 1):-1:1
        lateral = laterals[i]
        merged = [[lateral .+ interpolate_bilinear(first(merged), size(lateral)[1:2])];
                  merged]
    end
    return merged
end

function (m::UperNetHead)(features)
    check_features(features, length(m.lateral_convs) + 1)
    laterals = [[conv(features[i]) for (i, conv) in enumerate(m.lateral_convs)];
                [psp_forward(m, last(features))]]
    merged = top_down(laterals)
    n = length(m.fpn_convs)
    fpn_outs = [i <= n ? m.fpn_convs[i](merged[i]) : merged[i] for i in eachindex(merged)]
    target = size(first(fpn_outs))[1:2]
    fused = cat_channels([interpolate_bilinear(o, target) for o in fpn_outs])
    return m.classifier(m.fpn_bottleneck(fused))
end

"""
    UperNetFCNHead(in_channels; in_index = 3, kernel_size = 3, dilation = 1,
                   auxiliary_in_channels = nothing, auxiliary_channels = 256,
                   auxiliary_num_convs = 1, auxiliary_concat_input = false, num_labels)

transformers' `UperNetFCNHead`, the auxiliary head of UperNet, a fully convolutional
network on the feature map `in_index` (1-based: transformers' default `in_index = 2` is the
third map), of `in_channels[in_index]` channels unless `auxiliary_in_channels` is set:
`auxiliary_num_convs` [`UperNetConvModule`](@ref Metalhead.Transformers.UperNetConvModule)s
(`convs`), the input concatenated with their output and fused by `conv_cat` with
`auxiliary_concat_input`, and the `classifier`. It returns the logits
(W, H, num_labels, batch) at the size of that map. The keyword arguments `auxiliary_*` and
`num_labels` are the fields of `UperNetConfig`.
"""
@concrete struct UperNetFCNHead
    convs
    conv_cat
    classifier
    in_index::Int
end
Flux.@layer UperNetFCNHead

function UperNetFCNHead(in_channels; in_index::Integer = 3, kernel_size::Integer = 3,
        dilation::Integer = 1, auxiliary_in_channels = nothing,
        auxiliary_channels::Integer = 256, auxiliary_num_convs::Integer = 1,
        auxiliary_concat_input::Bool = false, num_labels::Integer)
    in_channels = collect(Int, in_channels)
    in_index in eachindex(in_channels) ||
        throw(ArgumentError("`in_index = $in_index` must select one of the " *
                            "$(length(in_channels)) feature maps"))
    auxiliary_num_convs >= 0 ||
        throw(ArgumentError("`auxiliary_num_convs` must not be negative"))
    c_in = something(auxiliary_in_channels, in_channels[in_index])
    c = auxiliary_channels
    check_head_args([c_in], c, num_labels)
    padding = (kernel_size ÷ 2) * dilation
    convs = auxiliary_num_convs == 0 ? identity :
            Chain([UperNetConvModule((i == 1 ? c_in : c) => c, kernel_size; padding,
                       dilation) for i in 1:auxiliary_num_convs]...)
    conv_cat = auxiliary_concat_input ?
               UperNetConvModule(c_in + c => c, kernel_size; padding = kernel_size ÷ 2) :
               nothing
    # with no convs, the classifier (of `auxiliary_channels`) takes the input itself, which
    # transformers builds but fails to run unless they have the same channels
    auxiliary_num_convs == 0 && c_in != c &&
        throw(ArgumentError("with `auxiliary_num_convs = 0`, the input's $c_in channels " *
                            "must be `auxiliary_channels = $c`"))
    return UperNetFCNHead(convs, conv_cat, init_layer(Conv, (1, 1), c => num_labels),
        Int(in_index))
end

function (m::UperNetFCNHead)(features)
    x = features[m.in_index]
    y = m.convs(x)
    m.conv_cat === nothing || (y = m.conv_cat(cat(x, y; dims = 3)))
    return m.classifier(y)
end
