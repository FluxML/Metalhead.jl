abstract type _MBConfig end

struct MBConvConfig <: _MBConfig
    kernel_size::Dims{2}
    inplanes::Integer
    outplanes::Integer
    expansion::Real
    stride::Integer
    nrepeats::Integer
end
function MBConvConfig(kernel_size::Integer, inplanes::Integer, outplanes::Integer,
                      expansion::Real, stride::Integer, nrepeats::Integer,
                      width_mult::Real = 1, depth_mult::Real = 1)
    inplanes = _round_channels(inplanes * width_mult, 8)
    outplanes = _round_channels(outplanes * width_mult, 8)
    nrepeats = ceil(Int, nrepeats * depth_mult)
    return MBConvConfig((kernel_size, kernel_size), inplanes, outplanes, expansion,
                        stride, nrepeats)
end

function efficientnetblock(m::MBConvConfig, norm_layer)
    layers = []
    explanes = _round_channels(m.inplanes * m.expansion, 8)
    push!(layers,
          mbconv(m.kernel_size, m.inplanes, explanes, m.outplanes, swish; norm_layer,
                 stride = m.stride, reduction = 4))
    explanes = _round_channels(m.outplanes * m.expansion, 8)
    append!(layers,
            [mbconv(m.kernel_size, m.outplanes, explanes, m.outplanes, swish; norm_layer,
                    stride = 1, reduction = 4) for _ in 1:(m.nrepeats - 1)])
    return Chain(layers...)
end

struct FusedMBConvConfig <: _MBConfig
    kernel_size::Dims{2}
    inplanes::Integer
    outplanes::Integer
    expansion::Real
    stride::Integer
    nrepeats::Integer
end
function FusedMBConvConfig(kernel_size::Integer, inplanes::Integer, outplanes::Integer,
                           expansion::Real, stride::Integer, nrepeats::Integer)
    return FusedMBConvConfig((kernel_size, kernel_size), inplanes, outplanes, expansion,
                             stride, nrepeats)
end

function efficientnetblock(m::FusedMBConvConfig, norm_layer)
    layers = []
    explanes = _round_channels(m.inplanes * m.expansion, 8)
    push!(layers,
          fused_mbconv(m.kernel_size, m.inplanes, explanes, m.outplanes, swish;
                       norm_layer, stride = m.stride))
    explanes = _round_channels(m.outplanes * m.expansion, 8)
    append!(layers,
            [fused_mbconv(m.kernel_size, m.outplanes, explanes, m.outplanes, swish;
                          norm_layer, stride = 1) for _ in 1:(m.nrepeats - 1)])
    return Chain(layers...)
end

function efficientnet(block_configs::AbstractVector{<:_MBConfig};
                      headplanes::Union{Nothing, Integer} = nothing,
                      norm_layer = BatchNorm, dropout_rate = nothing,
                      inchannels::Integer = 3, nclasses::Integer = 1000)
    layers = []
    # stem of the model
    append!(layers,
            conv_norm((3, 3), inchannels, block_configs[1].inplanes, swish; norm_layer,
                      stride = 2, pad = SamePad()))
    # building inverted residual blocks
    append!(layers, [efficientnetblock(cfg, norm_layer) for cfg in block_configs])
    # building last layers
    outplanes = block_configs[end].outplanes
    headplanes = isnothing(headplanes) ? outplanes * 4 : headplanes
    append!(layers,
            conv_norm((1, 1), outplanes, headplanes, swish; pad = SamePad()))
    return Chain(Chain(layers...), create_classifier(headplanes, nclasses; dropout_rate))
end
