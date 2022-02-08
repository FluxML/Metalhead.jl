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
    mlpblock(planes, expansion_factor = 4, dropout = 0., dense = Dense)

Feedforward block used in many vision transformer-like models.

# Arguments
- `planes`: Number of dimensions in the input and output.
- `hidden_planes`: Number of dimensions in the intermediate layer.
- `dropout`: Dropout rate.
- `dense`: Type of dense layer to use in the feedforward block.
- `activation`: Activation function to use.
"""
function mlpblock(planes, hidden_planes, dropout = 0., dense = Dense; activation = gelu)
  Chain(dense(planes, hidden_planes, activation), Dropout(dropout),
        dense(hidden_planes, planes, activation), Dropout(dropout))
end

"""
    Attention{T}

Self attention layer used by transformer models. Can be instantiated with a layer that produces
the key, value and query vectors from the input.
"""
struct Attention{T}
  qkv::T
end

Attention(in, out) = Attention(Dense(in, out * 3; bias = false))

@functor Attention

function (attn::Attention)(x::AbstractArray{T}) where T
  q, k, v = chunk(attn.qkv(x), 3; dim = 1)
  scale = convert(T, sqrt(size(q, 1)))
  score = softmax(batched_mul(batched_transpose(q), k) / scale)
  attention = batched_mul(v, score)

  return attention
end

struct MHAttention{Q <: Integer, S, T}
  nheads::Q
  heads::S
  projection::T
end

"""
    MHAttention(in, hidden, nheads, dropout = 0.)

Multi-head self-attention layer used in many vision transformer-like models.

# Arguments
- `in`: Number of dimensions in the input.
- `hidden`: Number of dimensions in the intermediate layer.
- `nheads`: Number of attention heads.
- `dropout`: Dropout rate for the projection layer.
"""
function MHAttention(in, hidden, nheads, dropout = 0.)
  project_out = !(nheads == 1 && hidden == in)
  inheads, innerheads = chunk(1:in, nheads), chunk(1:hidden, nheads)
  heads = Parallel(vcat, [Attention(length(i), length(o)) for (i, o) in zip(inheads, innerheads)]...)
  projection = project_out ? Chain(Dense(hidden, in), Dropout(dropout)) : identity

  MHAttention(nheads, heads, projection)
end

@functor MHAttention

function (mha::MHAttention)(x)
  xhead = chunk(x, mha.nheads; dim = 1)

  return mha.projection(mha.heads(xhead...))
end

"""
    Patching{T <: Integer}

Patching layer used by many vision transformer-like models to split the input image into patches.
Can be instantiated with a tuple `(patch_height, patch_width)` or a single value `patch_size`.
"""
struct Patching{T <: Integer}
  patch_height::T
  patch_width::T
end

Patching(patch_size) = Patching(patch_size, patch_size)

function (p::Patching)(x)
  h, w, c, n = size(x)
  hp, wp = h ÷ p.patch_height, w ÷ p.patch_width
  xpatch = reshape(x, hp, p.patch_height, wp, p.patch_width, c, n)

  return reshape(permutedims(xpatch, (1, 3, 5, 2, 4, 6)), p.patch_height * p.patch_width * c, 
                 hp * wp, n)
end

@functor Patching

"""
    PosEmbedding{T}

Positional embedding layer used by many vision transformer-like models. Instantiated with an 
embedding vector which is a learnable parameter.
"""
struct PosEmbedding{T}
  embedding_vector::T
end

(p::PosEmbedding)(x) = x .+ p.embedding_vector[:, 1:size(x)[2], :]

@functor PosEmbedding

"""
    CLSTokens{T}

Appends class tokens to the input that are used for classfication by many vision 
transformer-like models. Instantiated with a class token vector which is a learnable parameter.
"""
struct CLSTokens{T}
  cls_token::T
end

function(m::CLSTokens)(x)
  cls_tokens = repeat(m.cls_token, 1, 1, size(x)[3])
  return cat(cls_tokens, x; dims = 2)
end

@functor CLSTokens
