# Ported from timm.models.vision_transformer (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/vit_* and test/references/timm--vit_*.

@concrete struct ViTAttention
    num_heads
    qkv
    attn_drop
    proj
    proj_drop
end
Flux.@layer ViTAttention

function (m::ViTAttention)(x)
    _, n, b = size(x)
    qkv = reshape(m.qkv(x), :, m.num_heads, 3, n, b)
    q, k, v = qkv[:, :, 1, :, :], qkv[:, :, 2, :, :], qkv[:, :, 3, :, :]
    y = NNlib.scaled_dot_product_attention(q, k, v; fdrop = m.attn_drop)
    return m.proj_drop(m.proj(reshape(y, :, n, b)))
end

@concrete struct ViTBlock
    norm1
    attn
    drop_path1
    norm2
    mlp
    drop_path2
end
Flux.@layer ViTBlock

function (m::ViTBlock)(x)
    x = x .+ m.drop_path1(m.attn(m.norm1(x)))
    return x .+ m.drop_path2(m.mlp(m.norm2(x)))
end

@concrete struct VisionTransformer
    cls_token
    pos_embed
    patch_embed
    pos_drop
    blocks
    norm
    head_drop
    head
end
Flux.@layer VisionTransformer

function _pos_embed(m::VisionTransformer, x)
    x = cat(repeat(m.cls_token, 1, 1, size(x, 3)), x; dims = 2)
    return m.pos_drop(x .+ m.pos_embed)
end

function forward_features(m::VisionTransformer, x)
    x = _pos_embed(m, m.patch_embed(x))
    return m.norm(m.blocks(x))
end

function forward_head(m::VisionTransformer, x; pre_logits::Bool = false)
    x = m.head_drop(x[:, 1, :])
    return pre_logits ? x : m.head(x)
end

(m::VisionTransformer)(x) = forward_head(m, forward_features(m, x))

# timm's DropPath on (C, L, B) tokens: one mask per sample.
token_drop_path(p) = dropout_or_identity(p; dims = 3)

# timm's init_weights_vit_timm for nn.Linear
vit_dense(in_out; bias = true) = Dense(in_out; bias, init = TRUNC_NORMAL)

function vit_block(dim, num_heads; mlp_ratio, qkv_bias, proj_drop, attn_drop, drop_path)
    dim % num_heads == 0 ||
        error("`embed_dim` ($dim) must be divisible by `num_heads` ($num_heads)")
    hidden = floor(Int, dim * mlp_ratio)
    attn = ViTAttention(
        num_heads, vit_dense(dim => 3dim; bias = qkv_bias), dropout_or_identity(attn_drop),
        vit_dense(dim => dim), dropout_or_identity(proj_drop))
    mlp = Mlp(dim, hidden; drop = proj_drop)
    return ViTBlock(LayerNorm(dim; eps = 1e-6), attn, token_drop_path(drop_path),
        LayerNorm(dim; eps = 1e-6), mlp, token_drop_path(drop_path))
end

"""
    vit(; img_size = 224, patch_size = 16, embed_dim = 768, depth = 12, num_heads = 12, kwargs...)

Port of timm's `VisionTransformer`: class token, learned position embedding, pre-norm blocks and
token pooling. Keyword arguments are those of the timm class; options other than the plain ViT
ones (register tokens, `qk_norm`, `init_values`, average pooling, dynamic image size, ...) error.
`img_size` and `patch_size` are integers or `(height, width)` tuples, as in timm.
"""
function vit(; img_size = 224, patch_size = 16, in_chans = 3,
        num_classes = 1000, global_pool = "token",
        embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4.0, qkv_bias = true,
        qk_norm = false, scale_attn_norm = false, scale_mlp_norm = false, proj_bias = true,
        init_values = nothing, class_token = true, pos_embed = "learn", no_embed_class = false,
        reg_tokens = 0, pre_norm = false, final_norm = true, fc_norm = nothing,
        pool_include_prefix = false, dynamic_img_size = false, dynamic_img_pad = false,
        drop_rate = 0.0, pos_drop_rate = 0.0, patch_drop_rate = 0.0, proj_drop_rate = 0.0,
        attn_drop_rate = 0.0, drop_path_rate = 0.0, weight_init = "", fix_init = false,
        embed_layer = nothing, embed_norm_layer = nothing, norm_layer = nothing,
        act_layer = nothing, block_fn = nothing, mlp_layer = nothing, attn_layer = nothing)
    check_unsupported("VisionTransformer"; global_pool = (global_pool, "token"),
        qk_norm = (qk_norm, false),
        scale_attn_norm = (scale_attn_norm, false), scale_mlp_norm = (
            scale_mlp_norm, false),
        proj_bias = (proj_bias, true), init_values = (init_values, nothing),
        class_token = (class_token, true), pos_embed = (pos_embed, "learn"),
        no_embed_class = (no_embed_class, false), reg_tokens = (reg_tokens, 0),
        pre_norm = (pre_norm, false), final_norm = (final_norm, true), fc_norm = (
            fc_norm, nothing),
        pool_include_prefix = (pool_include_prefix, false),
        dynamic_img_size = (dynamic_img_size, false), dynamic_img_pad = (
            dynamic_img_pad, false),
        patch_drop_rate = (patch_drop_rate, 0.0), weight_init = (weight_init, ""),
        fix_init = (fix_init, false), embed_layer = (embed_layer, nothing),
        embed_norm_layer = (embed_norm_layer, nothing), norm_layer = (norm_layer, nothing),
        act_layer = (act_layer, nothing), block_fn = (block_fn, nothing),
        mlp_layer = (mlp_layer, nothing), attn_layer = (attn_layer, nothing))
    ph, pw = to_2tuple(patch_size)
    ih, iw = to_2tuple(img_size)
    num_patches = (ih ÷ ph) * (iw ÷ pw)
    patch_embed = PatchEmbed(Conv((pw, ph), in_chans => embed_dim; stride = (pw, ph)))
    drop_path_rates = depth == 1 ? [0.0] : range(0, drop_path_rate; length = depth)
    blocks = Chain([vit_block(embed_dim, num_heads; mlp_ratio,
                        qkv_bias, proj_drop = proj_drop_rate,
                        attn_drop = attn_drop_rate, drop_path = dpr)
                    for dpr in drop_path_rates])
    head = num_classes > 0 ? vit_dense(embed_dim => num_classes) : identity
    return VisionTransformer(1.0f-6 .* randn(Float32, embed_dim, 1, 1),
        Flux.truncated_normal(; std = 0.02)(embed_dim, num_patches + 1, 1),
        patch_embed, dropout_or_identity(pos_drop_rate), blocks,
        LayerNorm(embed_dim; eps = 1e-6), dropout_or_identity(drop_rate), head)
end

const VIT_REFERENCES = Dict(
    "vit_tiny_patch16_224" =>
        ["timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k" => "7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106"],
    "vit_base_patch16_224" =>
        ["timm/vit_base_patch16_224.augreg2_in21k_ft_in1k" => "063c6c38a5d8510b2e57df480445e94b231dad2c"]
)

for (name, args) in [
    "vit_tiny_patch16_224" =>
        (; patch_size = 16, embed_dim = 192, depth = 12, num_heads = 3),
    "vit_small_patch32_224" =>
        (; patch_size = 32, embed_dim = 384, depth = 12, num_heads = 6),
    "vit_small_patch16_224" =>
        (; patch_size = 16, embed_dim = 384, depth = 12, num_heads = 6),
    "vit_base_patch32_224" =>
        (; patch_size = 32, embed_dim = 768, depth = 12, num_heads = 12),
    "vit_base_patch16_224" =>
        (; patch_size = 16, embed_dim = 768, depth = 12, num_heads = 12),
    # timm's only pretrained cfg of this architecture (orig_in21k) has no classifier, so create_model gives num_classes = 0
    "vit_large_patch32_224" =>
        (; patch_size = 32, embed_dim = 1024, depth = 24, num_heads = 16, num_classes = 0),
    "vit_large_patch16_224" =>
        (; patch_size = 16, embed_dim = 1024, depth = 24, num_heads = 16)
]
    register!(ArchSpec(
        :timm, name, vit; defaults = args, references = get(VIT_REFERENCES, name, [])))
end
