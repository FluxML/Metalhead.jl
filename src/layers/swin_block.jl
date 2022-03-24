struct swin_block
    norm1
    norm2
    window_size
    w_attn
    shift_size
    mlp
end

function swin_block(dim,num_heads,window_size,shift_size,qkv_bias=true,qk_scale=(dim//n_heads) ^ -0.5,mlp_ratio,drop=0.0,attn_drop=0.0,drop_path=0.0,act_layer=gelu,norm_layer=LayerNorm)
    window_size=window_size;
    shift_size=shift_size;
    norm1=norm_layer(dim);
    norm2=norm_layer(dim);
    w_attn=WindowAttention(window_size, dim, num_heads, qkv_bias, qk_scale, attn_drop, drop);
    mlp=mlp_block(dim,Int(dim*mlp_ratio);drop,Dense,act_layer);
    drop_path=DropPath(drop_path);
end

@functor swin_block

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
    attn_mask=get_attn_mask(sb.window_size,sb.shift_size,size(shortcut)[1],size(shortcut)[2]);
    attn_windows=sb.w_attn(x_windows,mask=attn_mask);
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

    

