"""
    MHAttention(nheads::Int, qkv_layer, attn_drop, projection)

Multi-head self-attention layer.

# Arguments:
- `nheads`: Number of heads
- `qkv_layer`: layer to be used for getting the query, key and value
- `attn_drop`: dropout rate after the self-attention layer
- `projection`: projection layer to be used after self-attention
"""
struct MHAttention{P, Q, R}
  nheads::Int
  qkv_layer::P
  attn_drop::Q
  projection::R
end

"""
    MHAttention(planes, nheads = 8; qkv_bias = false, attn_drop = 0., proj_drop = 0.)

Multi-head self-attention layer.

# Arguments:
- `planes`: number of input channels
- `nheads`: number of heads
- `qkv_bias`: whether to use bias in the layer to get the query, key and value
- `attn_drop`: dropout rate after the self-attention layer
- `proj_drop`: dropout rate after the projection layer
"""
function MHAttention(planes, nheads = 8; qkv_bias = false, attn_drop = 0., proj_drop = 0.)
  @assert planes % nheads == 0 "planes should be divisible by nheads"
  qkv_layer = Dense(planes, planes * 3; bias = qkv_bias)
  attn_drop = Dropout(attn_drop)
  proj = Chain(Dense(planes, planes), Dropout(proj_drop))

  MHAttention(nheads, qkv_layer, attn_drop, proj)
end

@functor MHAttention

function (m::MHAttention)(x::AbstractArray{T, 3}) where T
  features, len_seq, batch_size = size(x)
  q, k, v = chunk(reshape(m.qkv_layer(x), features ÷ m.nheads, m.nheads, len_seq, 3 * batch_size), 3; dims = 4)
  scale = convert(T, sqrt(size(q, 1) / m.nheads))
  attn = m.attn_drop(softmax(NeuralAttentionlib.matmul(q, permutedims(k, (2, 1, 3, 4))) * scale))
  x = m.projection(reshape(NeuralAttentionlib.matmul(attn, v), (features, len_seq, batch_size)))
end
