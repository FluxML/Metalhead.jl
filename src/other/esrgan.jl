using Flux
using Flux:@functor

function ConvBlock(inc,out,k,s,p,use_act)
    if use_act 
        return Chain(
            Conv((k,k),inc => out,stride = s,pad = p,bias=true),
            x -> leakyrelu.(x,0.2)
        )
    else
        return Chain(
            Conv((k,k),inc => out,stride = s,pad = p,bias=true)
        )
    end
end

function UpsampleBlock(inc,scale = 2)
    return Chain(
        Upsample(:nearest,scale = (scale,scale)),
        Conv((3,3),inc=>inc,stride = 1,pad = 1,bias=true),
        x -> leakyrelu.(x,0.2)
    )
end

mutable struct DenseResidualBlock
    residual_beta
    blocks
end

@functor DenseResidualBlock

function DenseResidualBlock(inc;c = 32,residual_beta = 0.2)
    blocks = []
    for i in 0:4
        in_channels = inc + c*i
        out_channels = i<=3 ? c : inc
        use_act = i<=3 ? true : false
        push!(blocks,ConvBlock(in_channels,out_channels,3,1,1,use_act))
    end

    return DenseResidualBlock(residual_beta,blocks)
end

function (m::DenseResidualBlock)(x) 
    new_inputs = x
    local out,new_inputs
    for block in m.blocks
        out = block(new_inputs)
        new_inputs = cat(new_inputs,out,dims=3)
    end
    return m.residual_beta * out + x
end

mutable struct RRDB
    residual_beta
    rrdb
end

@functor RRDB

function RRDB(inc;residual_beta = 0.2)
    rrdb = Chain([DenseResidualBlock(inc) for _ in 1:3]...)
    RRDB(residual_beta,rrdb)
end

(m::RRDB)(x) = m.rrdb(x)*m.residual_beta + x

mutable struct Generator
    initial
    residuals
    conv
    upsamples
    final
end

@functor Generator

function Generator(inc=3,nc=64,nb=23)
    initial = Conv((3,3),inc=>nc,stride = 1,pad = 1,bias=true)
    residuals = Chain([RRDB(nc) for _ in 1:nb]...)
    conv = Conv((3,3),nc=>nc,stride = 1,pad = 1)
    upsamples = Chain(UpsampleBlock(nc),UpsampleBlock(nc))
    final = Chain(
        Conv((3,3),nc=>nc,stride = 1,pad = 1,bias = true),
        x -> leakyrelu.(x,0.2),
        Conv((3,3),nc=>inc,stride = 1,pad = 1,bias=true)
    )
    Generator(initial,residuals,conv,upsamples,final)
end

function (m::Generator)(x)
    initial = m.initial(x)
    x = m.conv(m.residuals(initial)) + initial
    x = m.upsamples(x)
    x = m.final(x)
    return x
end

mutable struct Discriminator
    blocks
    classifier
end

@functor Discriminator

function Discriminator(;in_c = 3,features = [64, 64, 128, 128, 256, 256, 512, 512])
    blocks = []
    for (idx,feature) in enumerate(features)
        s = 1 + (idx - 1) % 2
        push!(blocks,ConvBlock(in_c,feature,3,s,1,true))
        in_c = feature
    end
    blocks = Chain(blocks...)
    classifier = Chain(
        AdaptiveMeanPool((6,6)),
        flatten,
        Dense(512 * 6 * 6, 1024),
        x -> leakyrelu.(x,0.2),
        Dense(1024,1)
    )
    Discriminator(blocks,classifier)
end

function (m::Discriminator)(x)
    x = m.blocks(x)
    x = m.classifier(x)
    return x
end