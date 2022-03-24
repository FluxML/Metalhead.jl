struct swin_block
    img_size
    norm1
    norm2
    window_size
    w_attn
    shift_size
    mlp
end

function swin_block(dim,img_size,num_heads,window_size,shift_size,qkv_bias=true,qk_scale=(dim//n_heads) ^ -0.5,mlp_ratio,drop=0.0,attn_drop=0.0,drop_path=0.0,act_layer=gelu,norm_layer=LayerNorm)
    window_size=window_size;
    img_size=img_size;
    norm1=norm_layer(dim);
    norm2=norm_layer(dim);
    if min(img_size)<=window_size
        shift_size=0;
        window_size=min(img_size);
    else
        shift_size=shift_size;
    end
    @assert 0<=shift_size<window_size "shift_size too large!"
    w_attn=WindowAttention(window_size, dim, num_heads, qkv_bias, qk_scale, attn_drop, drop);
    mlp=mlp_block(dim,Int(dim*mlp_ratio);drop,Dense,act_layer);
    drop_path=DropPath(drop_path);
end

@functor swin_block (norm1,norm2,w_attn,mlp)

function (sb::swin_block)(x)
    if sb.shift_size>0
        shifted_x=circshift(x,(-sb.shift_size,-sb.shift_size,0,0));
    else
        shifted_x=x
    end
    shortcut=x;
    x=sb.norm1(x);
    x_windows=window_partition(shifted_x,sb.window_size);
    x_windows=reshape(x_windows,sb.window_size[1]*sb.window_size[2],:,:);
    if shift_size> 0    
        attn_mask=get_attn_mask(sb.window_size,sb.shift_size,size(shortcut)[1],size(shortcut)[2]);
    else
        attn_mask=nothing;
    end
    attn_windows=sb.w_attn(x_windows,attn_mask);
    shifted_x=window_reverse(attn_windows,sb.window_size,size(shortcut)[1],size(shortcut)[2],size(shortcut)[3],size(shortcut)[4]);
    if sb.shift_size>0
        x=circshift(shifted_x,(sb.shift_size,sb.shift_size,0,0));
    else
        x=shifted_x;
    end
    x=reshape(x,size(shortcut)[1]*size(shortcut)[2],size(shortcut)[3],size(shortcut)[4]);
    x=broadcast(+,x,sb.drop_path(sb.mlp(sb.norm2(x))))
    return x
end

struct BasicLayer
    norm_layer
    downsample
    blocks
end

function BasicLayer(dim,img_size,depth,num_heads,window_size,mlp_ratio,qkv_bias=true,qk_scale=(dim//n_heads) ^ -0.5,drop=0.,attn_drop=0.,drop_path=0.,downsample=nothing)
    norm_layer=LayerNorm(dim);
    downsample=downsample(dim,img_size,norm_layer);
    shift_size_indicator(x)=floor(Int,iseven(x)*min(window_size)/2)
    seq=[swin_block(dim,img_size,num_heads,window_size,shift_size_indicator(i),qkv_bias,qk_scale,mlp_ratio,drop,attn_drop,drop_path,gelu,norm_layer) for i in 1:depth];
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
