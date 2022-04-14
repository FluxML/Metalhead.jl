struct BasicLayer
    norm_layer
    downsample
    blocks
end

function BasicLayer(dim,img_size,depth,num_heads,window_size,mlp_ratio,qkv_bias=true,qk_scale=(dim/num_heads) ^ -0.5,drop=0.,attn_drop=0.,drop_path=0.,downsample=nothing)
    norm_layer=LayerNorm(dim);
    downsample=downsample
    shift_size_indicator(x)=floor(Int,iseven(x)*minimum(window_size)/2)
    if typeof(drop_path)==Vector{Float64};
        seq=[Layers.swin_block(dim,img_size,num_heads,window_size,shift_size_indicator(i),qkv_bias,qk_scale,mlp_ratio,drop,attn_drop,drop_path[i],gelu,norm_layer) for i in 1:depth];
    else
        seq=[Layers.swin_block(dim,img_size,num_heads,window_size,shift_size_indicator(i),qkv_bias,qk_scale,mlp_ratio,drop,attn_drop,drop_path,gelu,norm_layer) for i in 1:depth];   
    end 
    blocks=Chain(seq...);
    BasicLayer(norm_layer,downsample,blocks);
end

@functor BasicLayer

function (b::BasicLayer)(x)
    if b.downsample===nothing
        return b.blocks(x)
    else
        return b.downsample(b.blocks(x))
    end
end
"""
`Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
https://arxiv.org/pdf/2103.14030
"""
struct SwinTransformer
    ape
    patchEmbed
    posEmbed
    layers
    norm
    meanpool
    head
end

function SwinTransformer(img_size::Tuple,patch_size::Tuple,window_size::Tuple,n_classes::Int,embed_dim::Int,depths=[2,2,6,2],num_heads=[3,6,12,24],mlp_ratio=4.,qkv_bias=true,qk_scale=nothing,drop_rate=0.0,attn_drop_rate=0.0,drop_path_rate=0.01,norm_layer=LayerNorm,ape=false, patch_norm=true)
    ape=ape;
    patchEmbed=Chain(Layers.PatchEmbedding(patch_size[1],patch_size[2]),Dropout(drop_rate));
    n_patches=convert(Int,(img_size[1]/patch_size[1])*(img_size[2]/patch_size[2]));
    n_feature=embed_dim*2^(length(depths)-1);
    init = ((embed_dim,n_patches))->rand(Distributions.truncated(Normal(0,0.02)),(embed_dim,n_patches));
    posEmbed=Chain(Layers.ViPosEmbedding(init(embed_dim,n_patches)),Dropout(drop_rate));
    sdpr=[x for x in 0:drop_path_rate/(sum(identity,depths)-1):drop_path_rate];
    seq_layers=[]
    sdpr_index=1
    if qk_scale===nothing
        for i in 1:length(depths)
            dim = Int(embed_dim * 2 ^ (i-1));
            downsample=Layers.PatchMerging(img_size,dim,norm_layer);
            push!(seq_layers,BasicLayer(dim,img_size,depths[i],num_heads[i],window_size,mlp_ratio,
            qkv_bias,(dim/num_heads[i])^-0.5,drop_rate,attn_drop_rate,sdpr[sdpr_index:sdpr_index+depths[i]-1],downsample));
            sdpr_index=sdpr_index+depths[i];
        end
    else
        for i in 1:length(depths)
            dim = Int(embed_dim * 2 ^ (i-1));
            downsample=Layers.PatchMerging(img_size,dim,norm_layer);
            push!(seq_layers,BasicLayer(dim,img_size,depths[i],num_heads[i],window_size,mlp_ratio,
            qkv_bias,qk_scale,drop_rate,attn_drop_rate,sdpr[sdpr_index:sdpr_index+depths[i]-1],downsample));
            sdpr_index=sdpr_index+depths[i];
        end
    end
    layers=Chain(seq_layers...);
    norm=LayerNorm(n_feature);
    meanpool=AdaptiveMeanPool((1,));
    head=Dense(n_feature,n_classes);
    SwinTransformer(ape,patchEmbed,posEmbed,layers,norm,meanpool,head);
end

@functor SwinTransformer (patchEmbed,posEmbed,layers,norm,head)

function (st::SwinTransformer)(x)
    x=st.patchEmbed(x);
    if st.ape
        x=broadcast(+,x,st.posEmbed);
    end
    x=Chain(st.layers,st.norm,st.meanpool)(x);
    x=flatten(x);
    return st.head(x)
end


