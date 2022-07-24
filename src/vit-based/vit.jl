# Attention layer used in ViT

"""
    MHAttention(planes::Integer, nheads::Integer = 8; qkv_bias::Bool = false, attn_dropout_rate = 0., proj_dropout_rate = 0.)

Multi-head self-attention layer.

# Arguments

  - `planes`: number of input channels
  - `nheads`: number of heads
  - `qkv_bias`: whether to use bias in the layer to get the query, key and value
  - `attn_dropout_rate`: dropout rate after the self-attention layer
  - `proj_dropout_rate`: dropout rate after the projection layer
"""
struct MHAttention{P, Q, R}
    nheads::Int
    qkv_layer::P
    attn_drop::Q
    projection::R
end
@functor MHAttention

function MHAttention(planes::Integer, nheads::Integer = 8; qkv_bias::Bool = false,
                     attn_dropout_rate = 0.0, proj_dropout_rate = 0.0)
    @assert planes % nheads==0 "planes should be divisible by nheads"
    qkv_layer = Dense(planes, planes * 3; bias = qkv_bias)
    attn_drop = Dropout(attn_dropout_rate)
    proj = Chain(Dense(planes, planes), Dropout(proj_dropout_rate))
    return MHAttention(nheads, qkv_layer, attn_drop, proj)
end

function (m::MHAttention)(x::AbstractArray{T, 3}) where {T}
    nfeatures, seq_len, batch_size = size(x)
    x_reshaped = reshape(x, nfeatures, seq_len * batch_size)
    qkv = m.qkv_layer(x_reshaped)
    qkv_reshaped = reshape(qkv, nfeatures ÷ m.nheads, m.nheads, seq_len, 3 * batch_size)
    query, key, value = chunk(qkv_reshaped, 3; dims = 4)
    scale = convert(T, sqrt(size(query, 1) / m.nheads))
    key_reshaped = reshape(permutedims(key, (2, 1, 3, 4)), m.nheads, nfeatures ÷ m.nheads,
                           seq_len * batch_size)
    query_reshaped = reshape(permutedims(query, (1, 2, 3, 4)), nfeatures ÷ m.nheads,
                             m.nheads, seq_len * batch_size)
    attention = m.attn_drop(softmax(batched_mul(query_reshaped, key_reshaped) .* scale))
    value_reshaped = reshape(permutedims(value, (1, 2, 3, 4)), nfeatures ÷ m.nheads,
                             m.nheads, seq_len * batch_size)
    pre_projection = reshape(batched_mul(attention, value_reshaped),
                             (nfeatures, seq_len, batch_size))
    y = m.projection(reshape(pre_projection, size(pre_projection, 1), :))
    return reshape(y, :, seq_len, batch_size)
end

# Block in the transformer in ViT

function vitblock(planes, nheads; mlp_ratio = 4.0, qkv_bias = false,
                  layerscale_init = 1.0f-5, norm_type = residualprenorm,
                  dropout_rate = 0.0, attn_dropout_rate = 0.0,
                  proj_dropout_rate = 0.0, drop_path_rate = 0.0,
                  activation = gelu, norm_layer = LayerNorm)
    @assert norm_type in (residualprenorm, residualpostnorm)
    "`norm_type` should be either `residualprenorm` or `residualpostnorm`"
    if norm_type == residualpostnorm
        layerscale_init = 0.0f0
        @info "Disabling LayerScale for `norm_type == postnorm`" maxlog=1
    end
    return Chain(norm_type(planes,
                           Chain(MHAttention(planes, nheads; qkv_bias, attn_dropout_rate,
                                             proj_dropout_rate),
                                 LayerScale(planes, layerscale_init),
                                 DropPath(drop_path_rate)); norm_layer),
                 norm_type(planes,
                           Chain(mlp_block(planes, Int(planes * mlp_ratio); dropout_rate,
                                           activation),
                                 LayerScale(planes, layerscale_init),
                                 DropPath(drop_path_rate)); norm_layer))
end

"""
    vit(imsize::Dims{2} = (256, 256); inchannels = 3, patch_size::Dims{2} = (16, 16),
        embedplanes = 768, depth = 6, nheads = 16, mlp_ratio = 4.0, dropout_rate = 0.1,
        emb_dropout_rate = 0.1, pool = :class, nclasses = 1000)

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
function vit(imsize::Dims{2} = (256, 256); patch_size::Dims{2} = (16, 16),
             block_fn = vitblock, use_cls_token = true, pre_cls_token = true,
             norm_type = residualprenorm, embedplanes = 768, depth = 12, nheads = 12,
             mlp_ratio = 4.0, layerscale_init = 1.0f-5, activation = gelu,
             norm_layer = LayerNorm, dropout_rate = 0.0, drop_path_rate = 0.0,
             pool = :class, inchannels = 3, nclasses = 1000)
    @assert pool in [:class, :mean]
    "Pool type must be either `:class` (class token) or `:mean` (mean pooling)"
    npatches = prod(imsize .÷ patch_size)
    emb_dropout_rate = attn_dropout_rate = proj_dropout_rate = dropout_rate
    dp_rates = linear_scheduler(drop_path_rate; depth)
    transformer = Chain([block_fn(embedplanes, nheads; mlp_ratio, qkv_bias = false,
                                  layerscale_init, norm_type,
                                  attn_dropout_rate, proj_dropout_rate,
                                  drop_path_rate = dp_rates[i],
                                  activation, norm_layer)
                         for i in eachindex(dp_rates)]...)
    pos_embed = []
    embedlen = pre_cls_token ? npatches + 1 : npatches
    push!(pos_embed, PositionalEmbedding(embedplanes, embedlen))
    if use_cls_token
        cls_token = ClassTokens(embedplanes)
        pre_cls_token ? pushfirst!(pos_embed, cls_token) : push!(pos_embed, cls_token)
    end
    pool_layer = pool == :class ? x -> x[:, 1, :] : seconddimmean
    fc = create_classifier(embedplanes, nclasses; activation = tanh, pool_layer,
                           dropout_rate)
    return Chain(Chain(PatchEmbedding(imsize; inchannels, patch_size, embedplanes),
                       Chain(pos_embed..., Dropout(emb_dropout_rate)),
                       transformer,
                       LayerNorm(embedplanes)), fc)
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
