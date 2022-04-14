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
  B, C, N = size(x)
  q, k, v = chunk(reshape(m.qkv_layer(x), B ÷ m.nheads, m.nheads, C, 3 * N), 3; dims = 4)
  scale = convert(T, sqrt(size(q, 1) / m.nheads))
  attn = m.attn_drop(softmax(NeuralAttentionlib.matmul(q, permutedims(k, (2, 1, 3, 4))) * scale))
  x = m.projection(reshape(NeuralAttentionlib.matmul(attn, v), (B, C, N)))
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

function WindowAttention(window_size,dim,n_heads;qkv_bias=true,qk_scale=(dim/n_heads) ^ -0.5,attn_drop=0.,drop=0.)
  relative_bias = reshape(get_relative_bias(window_size,n_heads),1,1,:);
  n_heads = n_heads;
  window_size = window_size;
  qk_scale = qk_scale;
  qkv_bias=qkv_bias;
  qkv = Dense(dim,dim*3,bias=qkv_bias);
  attn_drop = Dropout(attn_drop);
  proj = Dense(dim,dim);
  proj = Chain(proj,Dropout(drop));
  WindowAttention(relative_bias,qkv,qk_scale,n_heads,window_size,attn_drop,proj);
end

@functor WindowAttention (relative_bias,qkv,attn_drop,proj)

function (wa::WindowAttention)(x,mask=nothing)#x is a window partitioned data of size (window height, window width, channels, num_windows * batchsize)
  B, C, N = size(x)
  q, k, v = chunk(reshape(m.qkv_layer(x), B ÷ m.nheads, m.nheads, C, 3 * N), 3; dims = 4)
  attn = wa.attn_drop(softmax(NeuralAttentionlib.matmul(q, permutedims(k, (2, 1, 3, 4))) * wa.qk_scale));
  attn = broadcast(+ , attn , permutedims(wa.relative_bias,[3,4]);
  if mask===nothing
    attn=softmax(attn);
  else
    numW=size(mask)[1];
    attn=broadcast(+,reshape(attn,size(attn)[1]//numW,numW,wa.n_heads,wa.window_size[1]*wa.window_size[2],wa.window_size[1]*wa.window_size[2]),reshape(unsqueeze(unsqueeze(mask,2),1),1,1,1,:,:));
    attn=softmax(attn);
  end
  attn=wa.attn_drop(attn);
  attn = wa.proj(reshape(NeuralAttentionlib.matmul(attn, v), (B, C, N)))
  return attn
end