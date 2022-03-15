"""
    transformer_encoder(planes, depth, heads, headplanes, mlppanes; dropout = 0.)

Transformer as used in the base ViT architecture.
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments
- `planes`: number of input channels
- `depth`: number of attention blocks
- `heads`: number of attention heads
- `headplanes`: number of hidden channels per head
- `mlppanes`: number of hidden channels in the MLP block
- `dropout`: dropout rate
"""
function transformer_encoder(planes, depth, heads, mlpplanes; dropout = 0.)
  layers = [Chain(SkipConnection(prenorm(planes, MHAttention(planes, heads; attn_drop = dropout)), +),
                  SkipConnection(prenorm(planes, mlp_block(planes, mlpplanes; dropout)), +))
            for _ in 1:depth]
  Chain(layers...)
end

"""
    vit(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = (16, 16), planes = 1024, 
        depth = 6, heads = 16, mlppanes = 2048, headplanes = 64, dropout = 0.1, emb_dropout = 0.1, 
        pool = :class, nclasses = 1000)

Creates a Vision Transformer (ViT) model.
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments
- `imsize`: image size
- `inchannels`: number of input channels
- `patch_size`: size of the patches
- `embedplanes`: the number of channels after the patch embedding
- `depth`: number of blocks in the transformer
- `heads`: number of attention heads in the transformer
- `mlpplanes`: number of hidden channels in the MLP block in the transformer
- `headplanes`: number of hidden channels per head in the transformer
- `dropout`: dropout rate
- `emb_dropout`: dropout rate for the positional embedding layer
- `pool`: pooling type, either :class or :mean
- `nclasses`: number of classes in the output
"""
function vit(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = (16, 16),
             embedplanes = 768, depth = 6, heads = 16, mlpplanes = 2048, dropout = 0.1, 
             emb_dropout = 0.1, pool = :class, nclasses = 1000)

  @assert pool in [:class, :mean]
  "Pool type must be either :class (class token) or :mean (mean pooling)"
  npatches = prod(imsize .÷ patch_size)
  return Chain(Chain(PatchEmbedding(imsize; inchannels, patch_size, embedplanes),
                     ClassTokens(embedplanes),
                     ViPosEmbedding(embedplanes, npatches + 1),
                     Dropout(emb_dropout),
                     transformer_encoder(embedplanes, depth, heads, mlpplanes; dropout),
                     (pool == :class) ? x -> x[:, 1, :] : seconddimmean),
               Chain(LayerNorm(embedplanes), Dense(embedplanes, nclasses)))
end

struct ViT
  layers
end

"""
    ViT(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = (16, 16),
        embedplanes = 768, depth = 6, heads = 16, mlpplanes = 2048, headplanes = 64,
        dropout = 0.1, emb_dropout = 0.1, pool = :class, nclasses = 1000)

Creates a Vision Transformer (ViT) model.
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments
- `imsize`: image size
- `inchannels`: number of input channels
- `patch_size`: size of the patches
- `embedplanes`: the number of channels after the patch embedding
- `depth`: number of blocks in the transformer
- `heads`: number of attention heads in the transformer
- `mlpplanes`: number of hidden channels in the MLP block in the transformer
- `headplanes`: number of hidden channels per head in the transformer
- `dropout`: dropout rate
- `emb_dropout`: dropout rate for the positional embedding layer
- `pool`: pooling type, either :class or :mean
- `nclasses`: number of classes in the output

See also [`Metalhead.vit`](#).
"""
function ViT(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = (16, 16),
             embedplanes = 768, depth = 12, heads = 16, mlpplanes = 3072,
             dropout = 0.1, emb_dropout = 0.1, pool = :class, nclasses = 1000)

  layers = vit(imsize; inchannels, patch_size, embedplanes, depth, heads, mlpplanes,
               dropout, emb_dropout, pool, nclasses)

  ViT(layers)
end

(m::ViT)(x) = m.layers(x)

backbone(m::ViT) = m.layers[1]
classifier(m::ViT) = m.layers[2]

@functor ViT
