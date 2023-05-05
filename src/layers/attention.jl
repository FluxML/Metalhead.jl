"""
    MultiHeadSelfAttention(planes::Integer, nheads::Integer = 8; qkv_bias::Bool = false, 
                attn_dropout_prob = 0., proj_dropout_prob = 0.)

Multi-head self-attention layer.

# Arguments

  - `planes`: number of input channels
  - `nheads`: number of heads
  - `qkv_bias`: whether to use bias in the layer to get the query, key and value
  - `attn_dropout_prob`: dropout probability after the self-attention layer
  - `proj_dropout_prob`: dropout probability after the projection layer
"""
struct MultiHeadSelfAttention{P, Q, R}
    nheads::Int
    qkv_layer::P
    attn_drop::Q
    projection::R
end
@functor MultiHeadSelfAttention

function MultiHeadSelfAttention(planes::Integer, nheads::Integer = 8; qkv_bias::Bool = false,
                     attn_dropout_prob = 0.0, proj_dropout_prob = 0.0)
    @assert planes % nheads==0 "planes should be divisible by nheads"
    qkv_layer = Dense(planes, planes * 3; bias = qkv_bias)
    attn_drop = Dropout(attn_dropout_prob)
    proj = Chain(Dense(planes, planes), Dropout(proj_dropout_prob))
    return MultiHeadSelfAttention(nheads, qkv_layer, attn_drop, proj)
end

function (m::MultiHeadSelfAttention)(x::AbstractArray{<:Number, 3})
    qkv = m.qkv_layer(x)
    q, k, v = chunk(qkv, 3, dims = 1)
    y, α = NNlib.dot_product_attention(q, k, v; m.nheads, fdrop = m.attn_drop)
    y = m.projection(y)
    return y
end
