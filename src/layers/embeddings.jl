"""
    PatchEmbedding(patch_size)
    PatchEmbedding(patch_height, patch_width)

Patch embedding layer used by many vision transformer-like models to split the input image into patches.
"""
struct PatchEmbedding
  patch_height::Int
  patch_width::Int
end

PatchEmbedding(patch_size) = PatchEmbedding(patch_size, patch_size);
PatchEmbedding(patch_size::Tuple)=PatchEmbedding(patch_size[1],patch_size[2]);

function (p::PatchEmbedding)(x)
  h, w, c, n = size(x)
  hp, wp = h ÷ p.patch_height, w ÷ p.patch_width
  xpatch = reshape(x, hp, p.patch_height, wp, p.patch_width, c, n)

  return reshape(permutedims(xpatch, (1, 3, 5, 2, 4, 6)), p.patch_height * p.patch_width * c, 
                 hp * wp, n)
end

@functor PatchEmbedding

struct ProjectedPatchEmbedding
  patch_height::Int
  patch_width::Int
  proj
  norm_layer
end

function ProjectedPatchEmbedding(p_h,p_w,in_channel,embed_dim,norm_layer=nothing)
  patch_height=p_h;
  patch_width=p_w;
  proj=Conv((1,1),in_channel=>embed_dim,stride=(patch_height,patch_width),pad=SamePad())
  if norm_layer !== nothing
   norm_layer=LayerNorm(embed_dim);
  else
    norm_layer=norm_layer
  end
  ProjectedPatchEmbedding(patch_height,patch_width,proj,norm_layer)
end
@functor ProjectedPatchEmbedding (proj,norm_layer)
function (p::ProjectedPatchEmbedding)(x)
  h, w, c, n = size(x)
  hp, wp = h ÷ p.patch_height, w ÷ p.patch_width
  xpatch = reshape(x, hp, p.patch_height, wp, p.patch_width, c, n)

  xpatch=reshape(permutedims(xpatch, (1, 3, 5, 2, 4, 6)), p.patch_height * p.patch_width * c, 
                 hp * wp, n)
  xpatch=p.proj(xpatch);
  if p.norm_layer!==nothing
    xpatch=p.norm_layer(xpatch)
  else
  end
  return xpatch
end

"""
    ViPosEmbedding(embedsize, npatches; init = (dims) -> rand(Float32, dims))

Positional embedding layer used by many vision transformer-like models.
"""
struct ViPosEmbedding{T}
  vectors::T
end

ViPosEmbedding(embedsize, npatches; init = (dims::NTuple{2, Int}) -> rand(Float32, dims)) = 
  ViPosEmbedding(init((embedsize, npatches)))

(p::ViPosEmbedding)(x) = x .+ p.vectors

@functor ViPosEmbedding

"""
    ClassTokens(dim; init = Flux.zeros32)

Appends class tokens to an input with embedding dimension `dim` for use in many vision transformer models.
"""
struct ClassTokens{T}
  token::T
end

ClassTokens(dim::Integer; init = Flux.zeros32) = ClassTokens(init(dim, 1, 1))

function (m::ClassTokens)(x)
  tokens = repeat(m.token, 1, 1, size(x, 3))
  return hcat(tokens, x)
end

@functor ClassTokens

struct PatchMerging
  input_resolution::Tuple
  dim::Int
  norm
  reduction
end

function PatchMerging(input_resolution::Tuple, dim::Int, norm_layer=LayerNorm)#input_resolution returns (h,w), which is the img_size
  input_resolution=input_resolution;
  reduction = Dense(4*dim,2*dim,false);
  norm = LayerNorm(4*dim);
  PatchMerging(input_resolution,dim,norm,reduction);
end
@functor PatchMerging (reduction,norm)
function (pm::PatchMerging)(x)
  b=size(x)[4];
  h,w=pm.input_resolution;
  @assert iseven(h)&&iseven(w) "h,w are odd"
  c=size(x)[3];
  x=reshape(x,h,w,c,b);
  x1=x[1:2:end,1:2:end,:,:]
  x2=x[2:2:end,1:2:end,:,:]
  x3=x[1:2:end,2:2:end,:,:]
  x4=x[2:2:end,2:2:end,:,:]
  x=cat(x1,x2,x3,x4;dims=3)
  return pm.reduction(pm.norm(x)) 
end
