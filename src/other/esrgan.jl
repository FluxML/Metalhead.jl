
function UpsampleBlock(inc, scale = 2)
    return Chain(
        Upsample(:nearest, scale = (scale, scale)),
        Conv((3, 3), inc => inc, Base.Fix2(leakyrelu, 0.2), pad = 1)
    )
end

struct DenseResidualBlock
    residual_beta
    blocks
end

function DenseResidualBlock(inc; 
                            c = 32, residual_beta = 0.2)
    blocks = []
    for i in 0:4
        in_channels = inc + c * i
        out_channels = i<=3 ? c : inc
        act = i <= 3 ? Base.Fix2(leakyrelu, 0.2) : identity
        push!(blocks, Conv((3, 3), in_channels => out_channels, act, pad = 1))
    end

    return DenseResidualBlock(residual_beta, blocks)
end

@functor DenseResidualBlock

function (m::DenseResidualBlock)(x) 
    new_inputs = x
    local out, new_inputs
    for block in m.blocks
        out = block(new_inputs)
        new_inputs = cat(new_inputs, out, dims = 3)
    end
    return m.residual_beta * out + x
end

mutable struct ResidualinResidualDenseBlock
    residual_beta
    rrdb
end

@functor ResidualinResidualDenseBlock

function ResidualinResidualDenseBlock(inc;
              residual_beta = 0.2)
    rrdb = Chain([DenseResidualBlock(inc) for _ in 1:3]...)
    ResidualinResidualDenseBlock(residual_beta, rrdb)
end

(m::ResidualinResidualDenseBlock)(x) = m.rrdb(x) * m.residual_beta + x

# struct esrgan_generator
#     initial
#     residuals
#     conv
#     upsamples
#     final
# end

# function esrgan_generator(inc = 3, nc = 64, nb = 23)
#     initial = Conv((3, 3), inc => nc, pad = 1)
#     residuals = Chain([ResidualinResidualDenseBlock(nc) for _ in 1:nb]...)
#     conv = Conv((3, 3), nc => nc, pad = 1)
#     upsamples = Chain(UpsampleBlock(nc), UpsampleBlock(nc))
#     final = Chain(
#         Conv((3, 3), nc => nc, Base.Fix2(leakyrelu, 0.2), pad = 1),
#         Conv((3, 3), nc => inc, pad = 1)
#     )

#     esrgan_generator(initial, residuals, conv, upsamples, final)
# end

# @functor esrgan_generator

# function (m::esrgan_generator)(x)
#     initial = m.initial(x)
#     x = m.conv(m.residuals(initial)) + initial
#     x = m.upsamples(x)
#     x = m.final(x)
#     return x
# end

function esrgan_generator(inc = 3, nc = 64, nb = 23)
    initial = Conv((3, 3), inc => nc, pad = 1)
    residuals = Chain([ResidualinResidualDenseBlock(nc) for _ in 1:nb]...)
    conv = Conv((3, 3), nc => nc, pad = 1)
    upsamples = Chain(UpsampleBlock(nc), UpsampleBlock(nc))
    final = Chain(
        Conv((3, 3), nc => nc, Base.Fix2(leakyrelu, 0.2), pad = 1),
        Conv((3, 3), nc => inc, pad = 1)
    )
    return Chain(initial, SkipConnection(Chain(conv, residuals), +), upsamples, final)
end

function esrgan_discriminator(; in_c = 3, features = [64, 64, 128, 128, 256, 256, 512, 512])
    blocks = []
    for (idx, feature) in enumerate(features)
        s = 1 + (idx - 1) % 2
        push!(blocks, Conv((3, 3), in_c => feature, Base.Fix2(leakyrelu, 0.2), stride = s, pad = 1))
        in_c = feature
    end
    blocks = Chain(blocks...)
    classifier = Chain(
        AdaptiveMeanPool((6, 6)),
        MLUtils.flatten,
        Dense(512 * 6 * 6, 1024, Base.Fix2(leakyrelu, 0.2)),
        Dense(1024, 1)
    )
    return Chain(blocks, classifier)
end

function esrgan()
    return Chain(discriminator = esrgan_discriminator(), generator = esrgan_generator())
end

