include("utilities.jl")

# Utility function for applying LayerNorm before a block
prenorm(planes, fn) = Chain(fn, LayerNorm(planes))

struct MHAttention
  heads
  scale
  qkvlayer
  outlayer
end

"""
    MHAttention(planes; heads = 8, headplanes = 64, dropout = 0.)

Multi-head attention block as used in the base ViT architecture.
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments
- `planes`: number of input channels
- `heads`: number of attention heads
- `headplanes`: number of hidden channels per head
- `dropout`: dropout rate
"""
function MHAttention(planes; heads = 8, headplanes = 64, dropout = 0.)
  hidden_planes = headplanes * heads
  outproject = !(heads == 1 && headplanes == planes)
  
  to_qkv = Dense(planes, hidden_planes * 3; bias = false)
  to_out = outproject ? Chain(Dense(hidden_planes, planes), Dropout(dropout)) : identity

  MHAttention(heads, headplanes ^ -0.5, to_qkv, to_out)
end

function (m::MHAttention)(x)
  q, k, v = chunk(m.qkvlayer(x), 3; dim = 1)
  @cast q[h, b, d, n] := q[(h, d), n, b] h in 1:m.heads
  @cast k[h, b, d, n] := k[(h, d), n, b] h in 1:m.heads
  @cast v[h, b, d, n] := v[(h, d), n, b] h in 1:m.heads
  dots = batchmul(q, permutedims(k, (2, 1, 3, 4))) * m.scale
  attn = softmax(dots; dims = 3)
  out = batchmul(attn, v)
  @cast out[(h, d), n, b] := out[h, b, d, n] in 1:m.heads
  m.outlayer(out)
end

@functor MHAttention

"""
    Transformer(planes, depth, heads, headplanes, mlppanes, dropout = 0.)

Transformer as used in the base ViT architecture.
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments
- `planes`: number of input channels
- `depth`: number of layers
- `heads`: number of attention heads
- `headplanes`: number of hidden channels per head
- `mlppanes`: number of hidden channels in the MLP block
- `dropout`: dropout rate
"""
function Transformer(planes, depth, heads, headplanes, mlpplanes, dropout = 0.)
  layers = [Chain(SkipConnection(prenorm(planes, MHAttention(planes; heads, headplanes, dropout)), +),
                  SkipConnection(prenorm(planes, mlpblock(planes, mlpplanes, dropout)), +)) 
            for _ in 1:depth]

  Chain(layers...)
end

"""
    vit(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = (16, 16), planes = 1024, 
        depth = 6, heads = 16, mlppanes = 2048, headplanes = 64, dropout = 0.1, emb_dropout = 0.1, 
        pool = "cls", nclasses = 1000)

Creates a Vision Transformer model as detailed in the paper An Image is Worth 16x16 
Words: Transformers for Image Recognition at Scale .
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments
- `imsize`: image size
- `inchannels`: number of input channels
- `patch_size`: size of the patches
- `planes`: the number of channels fed into the main model
- `depth`: number of blocks in the transformer
- `heads`: number of attention heads in the transformer
- `mlpplanes`: number of hidden channels in the MLP block in the transformer
- `headplanes`: number of hidden channels per head in the transformer
- `dropout`: dropout rate
- `emb_dropout`: dropout rate for the positional embedding layer
- `pool`: pooling type, either "cls" or "avg"
- `nclasses`: number of classes in the output
"""
function vit(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = (16, 16), planes = 1024, 
  depth = 6, heads = 16, mlppanes = 2048, headplanes = 64, dropout = 0.1, emb_dropout = 0.1, 
  pool = "cls", nclasses = 1000)

  im_height, im_width = imsize
  patch_height, patch_width = patch_size

  @assert (im_height % patch_height == 0) && (im_width % patch_width == 0)
  "Image dimensions must be divisible by the patch size."
  @assert pool in ["cls", "avg"]
  "Pool type must be either cls (cls token) or avg (mean pooling)"
  
  num_patches = (im_height ÷ patch_height) * (im_width ÷ patch_width)
  patchplanes = inchannels * patch_height * patch_width

  return Chain(Patching(patch_height, patch_width),
               Dense(patchplanes, planes),
               CLSTokens(rand(Float32, (planes, 1, 1))),
               PosEmbedding(rand(Float32, (planes, num_patches + 1, 1))),
               Dropout(emb_dropout),
               Transformer(planes, depth, heads, headplanes, mlppanes, dropout),
               (pool == "cls") ? x -> x[:, 1, :] : x -> _seconddimmean(x),
               Chain(LayerNorm(planes), Dense(planes, nclasses)))
end

struct ViT
  layers
end

"""
    ViT(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = (16, 16), planes = 1024, 
        depth = 6, heads = 16, mlppanes = 2048, headplanes = 64, dropout = 0.1, emb_dropout = 0.1, 
        pool = "cls", nclasses = 1000)

Creates a Vision Transformer model as detailed in the paper An Image is Worth 16x16 
Words: Transformers for Image Recognition at Scale .
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments
- `imsize`: image size
- `inchannels`: number of input channels
- `patch_size`: size of the patches
- `planes`: the number of channels fed into the main model
- `depth`: number of blocks in the transformer
- `heads`: number of attention heads in the transformer
- `mlpplanes`: number of hidden channels in the MLP block in the transformer
- `headplanes`: number of hidden channels per head in the transformer
- `dropout`: dropout rate
- `emb_dropout`: dropout rate for the positional embedding layer
- `pool`: pooling type, either "cls" or "avg"
- `nclasses`: number of classes in the output
"""
function ViT(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = (16, 16), planes = 1024, 
  depth = 6, heads = 16, mlppanes = 2048, headplanes = 64, dropout = 0.1, emb_dropout = 0.1, 
  pool = "cls", nclasses = 1000)
  
  layers = vit(imsize; inchannels, patch_size, planes, depth, heads, mlppanes, headplanes, 
               dropout, emb_dropout, pool, nclasses)

  ViT(layers)
end

(m::ViT)(x) = m.layers(x)

backbone(m::ViT) = m.layers[1:end-1]
classifier(m::ViT) = m.layers[end]

@functor ViT
