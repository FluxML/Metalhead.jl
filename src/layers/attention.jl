struct MHAttention{A, B, C}
  nheads::Integer
  projection::A
  attn_drop::B
  qkv_layer::C
  scale::Number
end

function MHAttention(planes, nheads = 8; qkv_bias = false, attn_drop = 0., proj_drop = 0.)
  @assert planes % nheads == 0 "planes should be divisible by nheads"
  scale = sqrt(planes / nheads)
  qkv = Dense(planes, planes * 3; bias = qkv_bias)
  attn_drop = Dropout(attn_drop)
  proj = Chain(Dense(planes, planes), Dropout(proj_drop))

  MHAttention(nheads, proj, attn_drop, qkv, scale)
end

@functor MHAttention

function (m::MHAttention)(x::AbstractArray{T}) where T
  B, C, N = size(x)
  qkv = reshape(m.qkv_layer(x), B ÷ m.nheads, m.nheads, C, N, 3)
  q, k, v = map(dropdims $ (; dims = 5), chunk(qkv, 3; dims = 5))
  attn = NeuralAttentionlib.unwrap_collapse(NeuralAttentionlib.matmul(q,
          permutedims(k, (2, 1, 3, 4)))) * convert(T, m.scale)
  attn = m.attn_drop(softmax(attn))
  x = NeuralAttentionlib.unwrap_collapse(NeuralAttentionlib.matmul(attn, v))
  x = reshape(x, (B, C, N))
  x = m.projection(x)
end
