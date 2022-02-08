# Utility function for applying LayerNorm before a block
prenorm(planes, fn) = Chain(fn, LayerNorm(planes))

"""
    transformer_encoder(planes, depth, heads, headplanes, mlppanes, dropout = 0.)

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
function transformer_encoder(planes, depth, heads, headplanes, mlpplanes, dropout = 0.)
  layers = [Chain(SkipConnection(prenorm(planes, MHAttention(planes, headplanes, heads; dropout)), +),
                  SkipConnection(prenorm(planes, mlpblock(planes, mlpplanes, dropout)), +)) 
            for _ in 1:depth]

  Chain(layers...)
end

"""
    vit(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = (16, 16), planes = 1024, 
        depth = 6, heads = 16, mlppanes = 2048, headplanes = 64, dropout = 0.1, emb_dropout = 0.1, 
        pool = :class, nclasses = 1000)

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
- `pool`: pooling type, either :class or :mean
- `nclasses`: number of classes in the output
"""
function vit(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = (16, 16), planes = 1024, 
             depth = 6, heads = 16, mlppanes = 2048, headplanes = 64, dropout = 0.1, emb_dropout = 0.1, 
             pool = :class, nclasses = 1000)

  im_height, im_width = imsize
  patch_height, patch_width = patch_size

  @assert (im_height % patch_height == 0) && (im_width % patch_width == 0)
  "Image dimensions must be divisible by the patch size."
  @assert pool in [:class, :mean]
  "Pool type must be either :class (class token) or :mean (mean pooling)"
  
  npatches = (im_height ÷ patch_height) * (im_width ÷ patch_width)
  patchplanes = inchannels * patch_height * patch_width

  return Chain(Chain(PatchEmbedding(patch_height, patch_width),
                     Dense(patchplanes, planes),
                     ClassTokens(planes),
                     ViPosEmbedding(planes, npatches + 1),
                     Dropout(emb_dropout),
                     transformer_encoder(planes, depth, heads, headplanes, mlppanes, dropout),
                     (pool == :class) ? x -> x[:, 1, :] : x -> _seconddimmean(x)),
               Chain(LayerNorm(planes), Dense(planes, nclasses)))
end

struct ViT
  layers
end

"""
    ViT(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = (16, 16), planes = 1024, 
        depth = 6, heads = 16, mlppanes = 2048, headplanes = 64, dropout = 0.1, emb_dropout = 0.1, 
        pool = "cls", nclasses = 1000)

Creates a Vision Transformer (ViT) model.
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
- `pool`: pooling type, either :class or :mean
- `nclasses`: number of classes in the output
"""
function ViT(imsize::NTuple{2} = (256, 256); inchannels = 3, patch_size = (16, 16), planes = 1024, 
  depth = 6, heads = 16, mlppanes = 2048, headplanes = 64, dropout = 0.1, emb_dropout = 0.1, 
  pool = :class, nclasses = 1000)
  
  layers = vit(imsize; inchannels, patch_size, planes, depth, heads, mlppanes, headplanes, 
               dropout, emb_dropout, pool, nclasses)

  ViT(layers)
end

(m::ViT)(x) = m.layers(x)

backbone(m::ViT) = m.layers[1]
classifier(m::ViT) = m.layers[2]

@functor ViT
