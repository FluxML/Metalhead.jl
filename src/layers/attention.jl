"""
    Attention(in => out)
    Attention(qkvlayer)

Self attention layer used by transformer models. Specify the `in` and `out` dimensions,
or directly provide a `qkvlayer` that maps an input the queries, keys, and values.
"""
struct Attention{T}
  qkv::T
end

Attention(dims::Pair{Int, Int}) = Attention(Dense(dims.first, dims.second * 3; bias = false))

@functor Attention

function (attn::Attention)(x::AbstractArray{T, 3}) where T
  q, k, v = chunk(attn.qkv(x), 3, dims = 1)
  scale = convert(T, sqrt(size(q, 1)))
  score = softmax(batched_mul(batched_transpose(q), k) / scale)
  attention = batched_mul(v, score)

  return attention
end

struct MHAttention{S, T}
  heads::S
  projection::T
end

"""
    MHAttention(in, hidden, nheads; dropout = 0.0)

Multi-head self-attention layer used in many vision transformer-like models.

# Arguments
- `in`: Number of dimensions in the input.
- `hidden`: Number of dimensions in the intermediate layer.
- `nheads`: Number of attention heads.
- `dropout`: Dropout rate for the projection layer.
"""
function MHAttention(in, hidden, nheads; dropout = 0.)
  if (nheads == 1 && hidden == in)
    return Attention(in => in)
  end
  inheads, innerheads = chunk(1:in, nheads), chunk(1:hidden, nheads)
  heads = Parallel(vcat, [Attention(length(i) => length(o)) for (i, o) in zip(inheads, innerheads)]...)
  projection = Chain(Dense(hidden, in), Dropout(dropout))

  MHAttention(heads, projection)
end

@functor MHAttention

function (mha::MHAttention)(x)
  nheads = length(mha.heads.layers)
  xhead = chunk(x, nheads, dims = 1)
  return mha.projection(mha.heads(xhead...))
end