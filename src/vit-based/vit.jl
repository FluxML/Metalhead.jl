include("utilities.jl")

# Utility function for applying LayerNorm before a block
prenorm(planes, fn) = Chain(fn, LayerNorm(planes))

"""
      mlpblock(inplanes, hidden_planes, dropout=0.)
  
MLP block as used in the base ViT architecture.
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments
- `inplanes`: number of input channels
- `hidden_planes`: number of hidden channels
- `dropout`: dropout rate
"""
function mlpblock(inplanes, hidden_planes, dropout = 0.)
  Chain(Dense(inplanes, hidden_planes, gelu),
        Dropout(dropout),
        Dense(hidden_planes, inplanes),
        Dropout(dropout))
end

struct attention
  heads
  scale
  qkvlayer
  outlayer
end

"""
    attention(planes; heads = 8, headplanes = 64, dropout = 0.)

Multi-head attention block as used in the base ViT architecture.
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments
- `planes`: number of input channels
- `heads`: number of attention heads
- `headplanes`: number of hidden channels per head
- `dropout`: dropout rate
"""
function attention(planes; heads = 8, headplanes = 64, dropout = 0.)
  hidden_planes = headplanes * heads
  outproject = !(heads == 1 && headplanes == planes)
  
  to_qkv = Dense(planes, hidden_planes * 3; bias = false)
  to_out = outproject ? Chain(Dense(hidden_planes, planes), Dropout(dropout)) : identity

  attention(heads, headplanes ^ -0.5, to_qkv, to_out)
end

function (m::attention)(x)
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

@functor attention

struct Transformer
  layers
end

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
  layers = [Chain(SkipConnection(prenorm(planes, attention(planes; heads, headplanes, dropout)), +),
                  SkipConnection(prenorm(planes, mlpblock(planes, mlpplanes, dropout)), +)) for _ in 1:depth]

  Transformer(layers)
end

function (m::Transformer)(x)
  for (attn, fn) in m.layers
    x = attn(x)
    x = fn(x)
  end
  return x
end

@functor Transformer

struct ViT
  ph
  pw
  planes
  patchplanes
  pos_embedding
  cls_token
  dropout
  transformer
  pool
  mlp_head
end

"""
    ViT(; image_size = 256, patch_size = 32, planes = 1024, depth = 6, heads = 16, 
    mlppanes = 2048, headplanes = 64, dropout = 0.1, emb_dropout = 0.1, pool = "cls", 
    nclasses = 1000)

Creates a Vision Transformer model as detailed in the paper An Image is Worth 16x16 
Words: Transformers for Image Recognition at Scale .
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments
- `image_size`: size of the input image
- `patch_size`: size of the patches
- `planes`: number of input channels fed into the transformer
- `depth`: number of layers
- `heads`: number of attention heads
- `headplanes`: number of hidden channels per head
- `mlppanes`: number of hidden channels in the MLP block
- `dropout`: dropout rate for the transformer
- `emb_dropout`: dropout rate after the positional embedding is applied
- `pool`: pooling method to use. Can be one of `cls`, `avg`
- `nclasses`: number of classes
"""
function ViT(; image_size = 256, patch_size = 32, planes = 1024, depth = 6, heads = 16, 
  mlppanes = 2048, headplanes = 64, dropout = 0.1, emb_dropout = 0.1, pool = "cls", 
  nclasses = 1000)

  image_height, image_width = pair(image_size)
  patch_height, patch_width = pair(patch_size)
  @assert image_height % patch_height == 0 && image_width % patch_width == 0 
  "Image dimensions must be divisible by the patch size."
  @assert pool in ["cls", "avg"]
  "Pool type must be either cls (cls token) or avg (mean pooling)"
  
  num_patches = (image_height ÷ patch_height) * (image_width ÷ patch_width)
  patchplanes = 3 * patch_height * patch_width

  pos_embedding = rand(Float32, (planes, num_patches + 1, 1))
  cls_token = rand(Float32, (planes, 1, 1))

  drop = Dropout(emb_dropout)
  transformer = Transformer(planes, depth, heads, headplanes, mlppanes, dropout)
  mlp_head = Chain(LayerNorm(planes), Dense(planes, nclasses))

  ViT(patch_height, patch_width, planes, patchplanes, pos_embedding, cls_token, drop, transformer,
      pool, mlp_head)
end

function (m::ViT)(x)
  x = @cast x[(p1, p2, c), (h, w), b] := x[(h, p1), (w, p2), c, b] p1 in 1:m.ph, p2 in 1:m.pw
  x = Dense(m.patchplanes, m.planes)(x)
  cls_tokens = repeat(m.cls_token, 1, 1, size(x)[3])
  x = cat(cls_tokens, x; dims = 2)
  x = x .+ m.pos_embedding[:, 1:(size(x)[2]), :]
  x = m.dropout(x)
  x = m.transformer(x)
  x = m.pool == "avg" ? mean(x; dims = 2)[:, 1, :] : x[:, 1, :]
  x = m.mlp_head(x)
end

@functor ViT
