"""
transformer_encoder(planes, depth, nheads; mlp_ratio = 4.0, dropout_rate = 0.)

Transformer as used in the base ViT architecture.
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments

  - `planes`: number of input channels
  - `depth`: number of attention blocks
  - `nheads`: number of attention heads
  - `mlp_ratio`: ratio of MLP layers to the number of input channels
  - `dropout_rate`: dropout rate
"""
function transformer_encoder(planes, depth, nheads; mlp_ratio = 4.0, dropout_rate = 0.0)
    layers = [Chain(SkipConnection(prenorm(planes,
                                           MHAttention(planes, nheads;
                                                       attn_drop_rate = dropout_rate,
                                                       proj_drop_rate = dropout_rate)), +),
                    SkipConnection(prenorm(planes,
                                           mlp_block(planes, floor(Int, mlp_ratio * planes);
                                                     dropout_rate)), +))
              for _ in 1:depth]
    return Chain(layers)
end

"""
    vit(imsize::Dims{2} = (256, 256); inchannels = 3, patch_size::Dims{2} = (16, 16),
        embedplanes = 768, depth = 6, nheads = 16, mlp_ratio = 4.0, dropout_rate = 0.1,
        emb_drop_rate = 0.1, pool = :class, nclasses = 1000)

Creates a Vision Transformer (ViT) model.
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments

  - `imsize`: image size
  - `inchannels`: number of input channels
  - `patch_size`: size of the patches
  - `embedplanes`: the number of channels after the patch embedding
  - `depth`: number of blocks in the transformer
  - `nheads`: number of attention heads in the transformer
  - `mlpplanes`: number of hidden channels in the MLP block in the transformer
  - `dropout_rate`: dropout rate
  - `emb_dropout`: dropout rate for the positional embedding layer
  - `pool`: pooling type, either :class or :mean
  - `nclasses`: number of classes in the output
"""
function vit(imsize::Dims{2} = (256, 256); inchannels = 3, patch_size::Dims{2} = (16, 16),
             embedplanes = 768, depth = 6, nheads = 16, mlp_ratio = 4.0, dropout_rate = 0.1,
             emb_drop_rate = 0.1, pool = :class, nclasses = 1000)
    @assert pool in [:class, :mean]
    "Pool type must be either `:class` (class token) or `:mean` (mean pooling)"
    npatches = prod(imsize .÷ patch_size)
    return Chain(Chain(PatchEmbedding(imsize; inchannels, patch_size, embedplanes),
                       ClassTokens(embedplanes),
                       ViPosEmbedding(embedplanes, npatches + 1),
                       Dropout(emb_drop_rate),
                       transformer_encoder(embedplanes, depth, nheads; mlp_ratio,
                                           dropout_rate),
                       (pool == :class) ? x -> x[:, 1, :] : seconddimmean),
                 Chain(LayerNorm(embedplanes), Dense(embedplanes, nclasses, tanh_fast)))
end

vit_configs = Dict(:tiny => (depth = 12, embedplanes = 192, nheads = 3),
                   :small => (depth = 12, embedplanes = 384, nheads = 6),
                   :base => (depth = 12, embedplanes = 768, nheads = 12),
                   :large => (depth = 24, embedplanes = 1024, nheads = 16),
                   :huge => (depth = 32, embedplanes = 1280, nheads = 16),
                   :giant => (depth = 40, embedplanes = 1408, nheads = 16,
                              mlp_ratio = 48 // 11),
                   :gigantic => (depth = 48, embedplanes = 1664, nheads = 16,
                                 mlp_ratio = 64 // 13))

"""
    ViT(mode::Symbol = base; imsize::Dims{2} = (256, 256), inchannels = 3,
        patch_size::Dims{2} = (16, 16), pool = :class, nclasses = 1000)

Creates a Vision Transformer (ViT) model.
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments

  - `mode`: the model configuration, one of
    `[:tiny, :small, :base, :large, :huge, :giant, :gigantic]`
  - `imsize`: image size
  - `inchannels`: number of input channels
  - `patch_size`: size of the patches
  - `pool`: pooling type, either :class or :mean
  - `nclasses`: number of classes in the output

See also [`Metalhead.vit`](#).
"""
struct ViT
    layers::Any
end
@functor ViT

function ViT(mode::Symbol = :base; imsize::Dims{2} = (256, 256), inchannels = 3,
             patch_size::Dims{2} = (16, 16), pool = :class, nclasses = 1000)
    _checkconfig(mode, keys(vit_configs))
    kwargs = vit_configs[mode]
    layers = vit(imsize; inchannels, patch_size, nclasses, pool, kwargs...)
    return ViT(layers)
end

(m::ViT)(x) = m.layers(x)

backbone(m::ViT) = m.layers[1]
classifier(m::ViT) = m.layers[2]
