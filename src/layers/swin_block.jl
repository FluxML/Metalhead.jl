struct swin_block
    window_size
    window_mask
    window_attn
    shift_size
end

function swin_block(dim,num_heads,window_size,shift_size,mlp_ratio,drop=0.0,attn_drop=0.0,drop_path=0.0,act_layer=gelu,norm_layer=LayerNorm)
    window_size=window_size;
    
