"""
    conv_bn(kernelsize, inplanes, outplanes, activation = relu;
            rev = false,
            stride = 1, pad = 0, dilation = 1, groups = 1, [bias, weight, init],
            initβ = Flux.zeros32, initγ = Flux.ones32, ϵ = 1f-5, momentum = 1f-1)

Create a convolution + batch normalization pair with ReLU activation.

# Arguments
- `kernelsize`: size of the convolution kernel (tuple)
- `inplanes`: number of input feature maps
- `outplanes`: number of output feature maps
- `activation`: the activation function for the final layer
- `rev`: set to `true` to place the batch norm before the convolution
- `stride`: stride of the convolution kernel
- `pad`: padding of the convolution kernel
- `dilation`: dilation of the convolution kernel
- `groups`: groups for the convolution kernel
- `bias`, `weight`, `init`: initialization for the convolution kernel (see [`Flux.Conv`](#))
- `initβ`, `initγ`: initialization for the batch norm (see [`Flux.BatchNorm`](#))
- `ϵ`, `momentum`: batch norm parameters (see [`Flux.BatchNorm`](#))
"""
function conv_bn(kernelsize, inplanes, outplanes, activation = relu;
                 rev = false,
                 initβ = Flux.zeros32, initγ = Flux.ones32, ϵ = 1f-5, momentum = 1f-1,
                 kwargs...)
  layers = []

  if rev
    activations = (conv = activation, bn = identity)
    bnplanes = inplanes
  else
    activations = (conv = identity, bn = activation)
    bnplanes = outplanes
  end

  push!(layers, Conv(kernelsize, Int(inplanes) => Int(outplanes), activations.conv; kwargs...))
  push!(layers, BatchNorm(Int(bnplanes), activations.bn;
                          initβ = initβ, initγ = initγ, ϵ = ϵ, momentum = momentum))

  return rev ? reverse(layers) : layers
end

"""
    cat_channels(x, y)

Concatenate `x` and `y` along the channel dimension (third dimension).
Equivalent to `cat(x, y; dims=3)`.
Convenient binary reduction operator for use with `Parallel`.
"""
cat_channels(x, y) = cat(x, y; dims = 3)

"""
    skip_projection(inplanes, outplanes, downsample = false)

Create a skip projection
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: the number of output feature maps
- `downsample`: set to `true` to downsample the input
"""
skip_projection(inplanes, outplanes, downsample = false) = downsample ? 
  Chain(conv_bn((1, 1), inplanes, outplanes, identity; stride = 2, bias = false)...) :
  Chain(conv_bn((1, 1), inplanes, outplanes, identity; stride = 1, bias = false)...)

# array -> PaddedView(0, array, outplanes) for zero padding arrays
"""
    skip_identity(inplanes, outplanes[, downsample])

Create a identity projection
([reference](https://arxiv.org/abs/1512.03385v1)).

# Arguments:
- `inplanes`: the number of input feature maps
- `outplanes`: the number of output feature maps
- `downsample`: this argument is ignored but it is needed for compatibility with [`resnet`](#).
"""
function skip_identity(inplanes, outplanes)
  if outplanes > inplanes
    return Chain(MaxPool((1, 1), stride = 2),
                 y -> cat(y, zeros(eltype(y),
                                   size(y, 1),
                                   size(y, 2),
                                   outplanes - inplanes, size(y, 4)); dims = 3))
  else
    return identity
  end
end
skip_identity(inplanes, outplanes, downsample) = skip_identity(inplanes, outplanes)

"""
    mlpblock(planes, hidden_planes; dropout = 0., dense = Dense, activation = gelu)

Feedforward block used in many vision transformer-like models.

# Arguments
- `planes`: Number of dimensions in the input and output.
- `hidden_planes`: Number of dimensions in the intermediate layer.
- `dropout`: Dropout rate.
- `dense`: Type of dense layer to use in the feedforward block.
- `activation`: Activation function to use.
"""
function mlpblock(planes, hidden_planes; dropout = 0., dense = Dense, activation = gelu)
  Chain(dense(planes, hidden_planes, activation), Dropout(dropout),
        dense(hidden_planes, planes, activation), Dropout(dropout))
end

"""
    Attention(in => out)
    Attention(qkvlayer)

Self attention layer used by transformer models. Specify the `in` and `out` dimensions,
or directly provide a `qkvlayer` that maps an input the queries, keys, and values.
"""
struct Attention{T}
  qkv::T
end

Attention(dims::Pair{Int, Int}) = Attention(Dense(dims.first, dims.second * 3; bias = false))

@functor Attention

function (attn::Attention)(x::AbstractArray{T, 3}) where T
  q, k, v = chunk(attn.qkv(x), 3, dims = 1)
  scale = convert(T, sqrt(size(q, 1)))
  score = softmax(batched_mul(batched_transpose(q), k) / scale)
  attention = batched_mul(v, score)

  return attention
end

struct MHAttention{S, T}
  heads::S
  projection::T
end

"""
    MHAttention(in, hidden, nheads; dropout = 0.0)

Multi-head self-attention layer used in many vision transformer-like models.

# Arguments
- `in`: Number of dimensions in the input.
- `hidden`: Number of dimensions in the intermediate layer.
- `nheads`: Number of attention heads.
- `dropout`: Dropout rate for the projection layer.
"""
function MHAttention(in, hidden, nheads; dropout = 0.)
  if (nheads == 1 && hidden == in)
    return Attention(in => in)
  end
  inheads, innerheads = chunk(1:in, nheads), chunk(1:hidden, nheads)
  heads = Parallel(vcat, [Attention(length(i) => length(o)) for (i, o) in zip(inheads, innerheads)]...)
  projection = Chain(Dense(hidden, in), Dropout(dropout))

  MHAttention(heads, projection)
end

@functor MHAttention

function (mha::MHAttention)(x)
  nheads = length(mha.heads.layers)
  xhead = chunk(x, nheads, dims = 1)
  return mha.projection(mha.heads(xhead...))
end

"""
    PatchEmbedding(patch_size)
    PatchEmbedding(patch_height, patch_width)

Patch embedding layer used by many vision transformer-like models to split the input image into patches.
"""
struct PatchEmbedding
  patch_height::Int
  patch_width::Int
end

PatchEmbedding(patch_size) = PatchEmbedding(patch_size, patch_size)

function (p::PatchEmbedding)(x)
  h, w, c, n = size(x)
  hp, wp = h ÷ p.patch_height, w ÷ p.patch_width
  xpatch = reshape(x, hp, p.patch_height, wp, p.patch_width, c, n)

  return reshape(permutedims(xpatch, (1, 3, 5, 2, 4, 6)), p.patch_height * p.patch_width * c, 
                 hp * wp, n)
end

@functor PatchEmbedding

"""
    ViPosEmbedding(embedsize, npatches; init = (dims) -> rand(Float32, dims))

Positional embedding layer used by many vision transformer-like models.
"""
struct ViPosEmbedding{T}
  vectors::T
end

ViPosEmbedding(embedsize, npatches; init = (dims::NTuple{2, Int}) -> rand(Float32, dims)) = 
  ViPosEmbedding(init((embedsize, npatches)))

(p::ViPosEmbedding)(x) = x .+ p.vectors

@functor ViPosEmbedding

"""
    ClassTokens(dim; init = Flux.zeros32)

Appends class tokens to an input with embedding dimension `dim` for use in many vision transformer models.
"""
struct ClassTokens{T}
  token::T
end

ClassTokens(dim::Integer; init = Flux.zeros32) = ClassTokens(init(dim, 1, 1))

function (m::ClassTokens)(x)
  tokens = repeat(m.token, 1, 1, size(x, 3))
  return hcat(tokens, x)
end

@functor ClassTokens
