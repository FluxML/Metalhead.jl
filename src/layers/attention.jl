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

struct WindowAttention
  relative_bias
  qkv
  qk_scale
  n_heads
  window_size
  attn_drop
  proj
end

function WindowAttention(window_size,dim,n_heads,qkv_bias=true,qk_scale=(dim//n_heads) ^ -0.5,attn_drop=0.,drop=0.)
  relative_bias = get_relative_bias(window_size,n_heads);
  n_heads = n_heads;
  window_size = window_size;
  qk_scale = qk_scale;
  qkv_bias=qkv_bias;
  qkv = Dense(dim,dim*3,bias=qkv_bias);
  attn_drop = Dropout(attn_drop);
  proj = Dense(dim,dim);
  proj = Chain(proj,Dropout(drop));
end

@functor WindowAttention (relative_bias,)

function (wa::WindowAttention)(x,mask=nothing)#x is a window partitioned data of size (window height, window width, channels, num_windows * batchsize)
  q, k, v = chunk(wa.qkv(x), 3, dims = 1);
  attn = batched_mul(batched_transpose(q), k) .* wa.qk_scale;
  attn = broadcast(+ , attn , wa.relative_bias);
  if mask===nothing
    attn=softmax(attn);
  else
    edge=size(mask)[1];
    attn=reshape(attn,size(attn)[1]//edge,edge,wa.n_heads,wa.window_size[1]*wa.window_size[2],wa.window_size[1]*wa.window_size[2])+unsqueeze(unsqueeze(mask,2),1);
    attn=softmax(attn);
  end
  attn=wa.attn_drop(attn);
  attn=batched_mul(batched_transpose(attn),v);
  attn=wa.proj(attn);
  attn=wa.proj_drop(attn);
  return attn
end