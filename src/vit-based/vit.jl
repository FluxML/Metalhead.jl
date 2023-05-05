"""
    transformer_encoder(planes, depth, nheads; mlp_ratio = 4.0, dropout_prob = 0.)

Transformer as used in the base ViT architecture.
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments

  - `planes`: number of input channels
  - `depth`: number of attention blocks
  - `nheads`: number of attention heads
  - `mlp_ratio`: ratio of MLP layers to the number of input channels
  - `dropout_prob`: dropout probability
"""
function transformer_encoder(planes::Integer, depth::Integer, nheads::Integer;
                             mlp_ratio = 4.0, dropout_prob = 0.0, qkv_bias=false)
    layers = [Chain(SkipConnection(prenorm(planes,
                                            MultiHeadSelfAttention(planes, nheads;
                                                       qkv_bias,
                                                       attn_dropout_prob = dropout_prob,
                                                       proj_dropout_prob = dropout_prob)),
                                   +),
                    SkipConnection(prenorm(planes,
                                           mlp_block(planes, floor(Int, mlp_ratio * planes);
                                                     dropout_prob)), +))
              for _ in 1:depth]
    return Chain(layers)
end

"""
    vit(imsize::Dims{2} = (256, 256); inchannels::Integer = 3, patch_size::Dims{2} = (16, 16),
        embedplanes = 768, depth = 6, nheads = 16, mlp_ratio = 4.0, dropout_prob = 0.1,
        emb_dropout_prob = 0.1, pool = :class, nclasses::Integer = 1000)

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
  - `dropout_prob`: dropout probability
  - `emb_dropout`: dropout probability for the positional embedding layer
  - `pool`: pooling type, either :class or :mean
  - `nclasses`: number of classes in the output
"""
function vit(imsize::Dims{2} = (256, 256); inchannels::Integer = 3,
             patch_size::Dims{2} = (16, 16), embedplanes::Integer = 768,
             depth::Integer = 6, nheads::Integer = 16, mlp_ratio = 4.0, dropout_prob = 0.1,
             emb_dropout_prob = 0.1, pool::Symbol = :class, nclasses::Integer = 1000, 
             qkv_bias = false)
    @assert pool in [:class, :mean]
    "Pool type must be either `:class` (class token) or `:mean` (mean pooling)"
    npatches = prod(imsize .÷ patch_size)
    return Chain(Chain(PatchEmbedding(imsize; inchannels, patch_size, embedplanes),
                       ClassTokens(embedplanes),
                       ViPosEmbedding(embedplanes, npatches + 1),
                       Dropout(emb_dropout_prob),
                       transformer_encoder(embedplanes, depth, nheads; mlp_ratio,
                                           dropout_prob, qkv_bias),
                       pool === :class ? x -> x[:, 1, :] : seconddimmean),
                 Chain(LayerNorm(embedplanes), Dense(embedplanes, nclasses)))
end

const VIT_CONFIGS = Dict(:tiny => (depth = 12, embedplanes = 192, nheads = 3),
                         :small => (depth = 12, embedplanes = 384, nheads = 6),
                         :base => (depth = 12, embedplanes = 768, nheads = 12),
                         :large => (depth = 24, embedplanes = 1024, nheads = 16),
                         :huge => (depth = 32, embedplanes = 1280, nheads = 16),
                         :giant => (depth = 40, embedplanes = 1408, nheads = 16,
                                    mlp_ratio = 48 // 11),
                         :gigantic => (depth = 48, embedplanes = 1664, nheads = 16,
                                       mlp_ratio = 64 // 13))

"""
    ViT(config::Symbol = base; imsize::Dims{2} = (256, 256), inchannels::Integer = 3,
        patch_size::Dims{2} = (16, 16), pool = :class, nclasses::Integer = 1000)

Creates a Vision Transformer (ViT) model.
([reference](https://arxiv.org/abs/2010.11929)).

# Arguments

  - `config`: the model configuration, one of
    `[:tiny, :small, :base, :large, :huge, :giant, :gigantic]`
  - `imsize`: image size
  - `inchannels`: number of input channels
  - `patch_size`: size of the patches
  - `pool`: pooling type, either :class or :mean
  - `nclasses`: number of classes in the output

See also [`Metalhead.vit`](@ref).
"""
struct ViT
    layers::Any
end
@functor ViT

function ViT(config::Symbol; imsize::Dims{2} = (256, 256), patch_size::Dims{2} = (16, 16),
             pretrain::Bool = false, inchannels::Integer = 3, nclasses::Integer = 1000, 
             qkv_bias=false)
    _checkconfig(config, keys(VIT_CONFIGS))
    layers = vit(imsize; inchannels, patch_size, nclasses, qkv_bias, VIT_CONFIGS[config]...)
    if pretrain
        loadpretrain!(layers, string("vit", config))
    end
    return ViT(layers)
end

(m::ViT)(x) = m.layers(x)

backbone(m::ViT) = m.layers[1]
classifier(m::ViT) = m.layers[2]
