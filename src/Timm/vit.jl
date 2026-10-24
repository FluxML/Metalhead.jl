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
    y = NNlib.scaled_dot_product_attention(q, k, v; fdrop = attention_fdrop(m.attn_drop, q))
    return m.proj_drop(m.proj(reshape(y, :, n, b)))
end

@concrete struct ViTBlock
    norm1
    attn
    ls1
    drop_path1
    norm2
    mlp
    ls2
    drop_path2
end
Flux.@layer ViTBlock

function (m::ViTBlock)(x)
    x = x .+ m.drop_path1(m.ls1(m.attn(m.norm1(x))))
    return x .+ m.drop_path2(m.ls2(m.mlp(m.norm2(x))))
end

@concrete struct VisionTransformer
    cls_token
    reg_token
    pos_embed
    patch_embed
    pos_drop
    patch_drop
    norm_pre
    blocks
    norm
    attn_pool
    fc_norm
    head_drop
    head
    global_pool
    num_prefix_tokens
    no_embed_class
    pool_include_prefix
    dynamic_img_size
    grid_size
end
Flux.@layer VisionTransformer

function _prefix_tokens(m::VisionTransformer, b)
    cls = m.cls_token === nothing ? () : (repeat(m.cls_token, 1, 1, b),)
    reg = m.reg_token === nothing ? () : (repeat(m.reg_token, 1, 1, b),)
    return (cls..., reg...)
end

function _pos_embed(m::VisionTransformer, x)
    b = size(x)[end]
    prefix = _prefix_tokens(m, b)
    pos_embed = m.pos_embed
    if m.dynamic_img_size
        c, w, h, _ = size(x)
        if pos_embed !== nothing
            pos_embed = resample_abs_pos_embed(pos_embed, (h, w), m.grid_size;
                num_prefix_tokens = m.no_embed_class ? 0 : m.num_prefix_tokens)
        end
        x = reshape(x, c, w * h, b)
    end
    pos_embed === nothing && return cat(prefix..., x; dims = 2)
    x = m.no_embed_class ? cat(prefix..., x .+ pos_embed; dims = 2) :
        cat(prefix..., x; dims = 2) .+ pos_embed
    return m.pos_drop(x)
end

_embed(m::VisionTransformer, x) = m.norm_pre(m.patch_drop(_pos_embed(m, m.patch_embed(x))))

forward_features(m::VisionTransformer, x) = m.norm(m.blocks(_embed(m, x)))

function _pool(m::VisionTransformer, x)
    np = m.num_prefix_tokens
    if m.attn_pool !== nothing
        return m.attn_pool(m.pool_include_prefix ? x : x[:, (np + 1):end, :])
    end
    m.global_pool == "" && return x
    m.global_pool == "token" && return x[:, 1, :]
    x = m.pool_include_prefix ? x : x[:, (np + 1):end, :]
    m.global_pool == "avg" && return dropdims(mean(x; dims = 2); dims = 2)
    m.global_pool == "max" && return dropdims(maximum(x; dims = 2); dims = 2)
    return dropdims((maximum(x; dims = 2) .+ mean(x; dims = 2)) ./ 2; dims = 2)
end

function forward_head(m::VisionTransformer, x; pre_logits::Bool = false)
    x = m.head_drop(m.fc_norm(_pool(m, x)))
    return pre_logits ? x : m.head(x)
end

(m::VisionTransformer)(x) = forward_head(m, forward_features(m, x))

function forward_intermediates(m::VisionTransformer, x; indices = nothing,
        return_prefix_tokens::Bool = false, norm::Bool = false, stop_early::Bool = false,
        output_fmt = "NCHW", intermediates_only::Bool = false)
    check_output_fmt(output_fmt, ("NCHW", "NLC"))
    blocks = m.blocks.layers
    take, max_index = feature_take_indices(length(blocks), indices)
    grid = patch_grid(m.patch_embed, x)
    outs = stage_outputs(blocks, _embed(m, x), stop_early ? max_index : length(blocks))
    np = m.num_prefix_tokens
    intermediates = map(take) do i
        y = norm ? m.norm(outs[i]) : outs[i]
        spatial = y[:, (np + 1):end, :]
        spatial = output_fmt == "NCHW" ? tokens_to_nchw(spatial, grid) : spatial
        return return_prefix_tokens && np > 0 ? (spatial, y[:, 1:np, :]) : spatial
    end
    intermediates_only && return intermediates
    return m.norm(last(outs)), intermediates
end

function feature_info(m::VisionTransformer)
    dim = size(m.patch_embed.proj.weight, 4)
    reduction = maximum(m.patch_embed.proj.stride)
    return [(; num_chs = dim, reduction, module_name = "blocks.$(i - 1)")
            for i in eachindex(m.blocks.layers)]
end

# timm's init_weights_vit_timm for nn.Linear
vit_dense(in_out; bias = true) = Dense(in_out; bias, init = TRUNC_NORMAL)

const VIT_ACTIVATIONS = Dict{Any, Any}(nothing => gelu_erf, "gelu" => gelu_erf,
    "gelu_tanh" => gelu_tanh, "quick_gelu" => quick_gelu)

function vit_block(dim, num_heads; mlp_ratio, qkv_bias, proj_bias, init_values, proj_drop,
        attn_drop, drop_path, norm_eps, act_layer)
    dim % num_heads == 0 ||
        error("`embed_dim` ($dim) must be divisible by `num_heads` ($num_heads)")
    hidden = floor(Int, dim * mlp_ratio)
    attn = ViTAttention(
        num_heads, vit_dense(dim => 3dim; bias = qkv_bias), dropout_or_identity(attn_drop),
        vit_dense(dim => dim; bias = proj_bias), dropout_or_identity(proj_drop))
    mlp = Mlp(dim, hidden; act_layer, bias = proj_bias, drop = proj_drop)
    ls() = init_values === nothing ? identity : layer_scale(dim, init_values)
    return ViTBlock(LayerNorm(dim; eps = norm_eps), attn, ls(), token_drop_path(drop_path),
        LayerNorm(dim; eps = norm_eps), mlp, ls(), token_drop_path(drop_path))
end

"""
    vit(; img_size = 224, patch_size = 16, embed_dim = 768, depth = 12, num_heads = 12, kwargs...)

Port of timm's `VisionTransformer`, including the options of DINOv2 (`init_values`,
`reg_tokens`, `no_embed_class`), CLIP (`pre_norm`, `act_layer = "quick_gelu"`) and SigLIP
(`class_token = false`, `global_pool = "map"`), and `dynamic_img_size`. Keyword arguments are
those of the timm class, except that the epsilon of `norm_layer` is given as `norm_eps`
(timm's `partial(LayerNorm, eps = ...)`). `act_layer` is `"gelu"` (the default),
`"gelu_tanh"` or `"quick_gelu"`. `img_size` and `patch_size` are integers or
`(height, width)` tuples, as in timm. Other options (`qk_norm`, custom layers, ...) error.
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
        norm_eps = 1e-6, act_layer = nothing, block_fn = nothing, mlp_layer = nothing,
        attn_layer = nothing)
    check_unsupported("VisionTransformer"; qk_norm = (qk_norm, false),
        scale_attn_norm = (scale_attn_norm, false), scale_mlp_norm = (
            scale_mlp_norm, false),
        dynamic_img_pad = (dynamic_img_pad, false),
        patch_drop_rate = (patch_drop_rate, 0.0), weight_init = (weight_init, ""),
        fix_init = (fix_init, false), embed_layer = (embed_layer, nothing),
        embed_norm_layer = (embed_norm_layer, nothing), norm_layer = (norm_layer, nothing),
        block_fn = (block_fn, nothing), mlp_layer = (mlp_layer, nothing),
        attn_layer = (attn_layer, nothing))
    global_pool in ("", "avg", "avgmax", "max", "token", "map") ||
        error("`global_pool = $(repr(global_pool))` is not supported by the Metalhead port of VisionTransformer")
    class_token || global_pool != "token" ||
        error("`global_pool = \"token\"` needs `class_token = true`")
    pos_embed in ("", "none", "learn") || error("unknown `pos_embed = $(repr(pos_embed))`")
    haskey(VIT_ACTIVATIONS, act_layer) ||
        error("`act_layer = $(repr(act_layer))` is not supported by the Metalhead port of VisionTransformer")
    act = VIT_ACTIVATIONS[act_layer]
    use_fc_norm = fc_norm === nothing ? global_pool in ("avg", "avgmax", "max") : fc_norm
    num_prefix_tokens = class_token + reg_tokens
    ph, pw = to_2tuple(patch_size)
    ih, iw = to_2tuple(img_size)
    grid_size = (ih ÷ ph, iw ÷ pw)
    num_patches = prod(grid_size)
    patch_embed = PatchEmbed(
        Conv((pw, ph), in_chans => embed_dim; stride = (pw, ph), bias = !pre_norm);
        flatten = !dynamic_img_size)
    embed_len = no_embed_class ? num_patches : num_patches + num_prefix_tokens
    drop_path_rates = depth == 1 ? [0.0] : range(0, drop_path_rate; length = depth)
    blocks = Chain([vit_block(embed_dim, num_heads; mlp_ratio, qkv_bias, proj_bias,
                        init_values, proj_drop = proj_drop_rate, attn_drop = attn_drop_rate,
                        drop_path = dpr, norm_eps, act_layer = act)
                    for dpr in drop_path_rates])
    norm() = LayerNorm(embed_dim; eps = norm_eps)
    attn_pool = global_pool == "map" ?
                attention_pool_latent(embed_dim; num_heads, mlp_ratio, norm_eps,
        act_layer = act) : nothing
    head = num_classes > 0 ? vit_dense(embed_dim => num_classes) : identity
    return VisionTransformer(
        class_token ? 1.0f-6 .* randn(Float32, embed_dim, 1, 1) : nothing,
        reg_tokens > 0 ? 1.0f-6 .* randn(Float32, embed_dim, reg_tokens, 1) : nothing,
        pos_embed == "learn" ? Flux.truncated_normal(; std = 0.02)(embed_dim, embed_len, 1) :
        nothing,
        patch_embed, dropout_or_identity(pos_drop_rate), identity,
        pre_norm ? norm() : identity, blocks,
        final_norm && !use_fc_norm ? norm() : identity, attn_pool,
        final_norm && use_fc_norm ? norm() : identity, dropout_or_identity(drop_rate), head,
        global_pool, num_prefix_tokens, no_embed_class, pool_include_prefix,
        dynamic_img_size, grid_size)
end

const VIT_REFERENCES = Dict(
    "vit_tiny_patch16_224" =>
        ["timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k" => "7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106"],
    "vit_base_patch16_224" =>
        ["timm/vit_base_patch16_224.augreg2_in21k_ft_in1k" => "063c6c38a5d8510b2e57df480445e94b231dad2c"],
    "vit_small_patch14_dinov2" =>
        ["timm/vit_small_patch14_dinov2.lvd142m" => "4610ca143709d58a633b6397a74412c2c3842454"],
    "vit_small_patch14_reg4_dinov2" =>
        ["timm/vit_small_patch14_reg4_dinov2.lvd142m" => "c04b5193082a8d5b0c4856c7937384a48136c5de"],
    "vit_base_patch16_clip_224" =>
        ["timm/vit_base_patch16_clip_224.laion2b" => "fbadfdd1c0e903c58855b70ae04740b22da9d305"],
    "vit_base_patch32_clip_quickgelu_224" =>
        ["timm/vit_base_patch32_clip_quickgelu_224.metaclip_400m" => "a6964321961725c5a073f3b658b0cd32a8ec1a96"],
    "vit_base_patch16_siglip_224" =>
        ["timm/vit_base_patch16_siglip_224.v2_webli" => "4c3661e5ac879a276ddc5ddc6d3f0ecc78fd5d82"]
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
        (; patch_size = 16, embed_dim = 1024, depth = 24, num_heads = 16),
    "vit_small_patch8_224" =>
        (; patch_size = 8, embed_dim = 384, depth = 12, num_heads = 6, num_classes = 0),
    "vit_base_patch8_224" =>
        (; patch_size = 8, embed_dim = 768, depth = 12, num_heads = 12),
    # img_size and num_classes below come from the default pretrained cfg, as in timm
    "vit_tiny_patch16_384" =>
        (; patch_size = 16, embed_dim = 192, depth = 12, num_heads = 3, img_size = 384),
    "vit_small_patch32_384" =>
        (; patch_size = 32, embed_dim = 384, depth = 12, num_heads = 6, img_size = 384),
    "vit_small_patch16_384" =>
        (; patch_size = 16, embed_dim = 384, depth = 12, num_heads = 6, img_size = 384),
    "vit_base_patch32_384" =>
        (; patch_size = 32, embed_dim = 768, depth = 12, num_heads = 12, img_size = 384),
    "vit_base_patch16_384" =>
        (; patch_size = 16, embed_dim = 768, depth = 12, num_heads = 12, img_size = 384),
    "vit_large_patch32_384" =>
        (; patch_size = 32, embed_dim = 1024, depth = 24, num_heads = 16, img_size = 384),
    "vit_large_patch16_384" =>
        (; patch_size = 16, embed_dim = 1024, depth = 24, num_heads = 16, img_size = 384)
]
    register!(ArchSpec(
        TIMM, name, vit; family = "vision_transformer", defaults = args,
        references = get(VIT_REFERENCES, name, [])))
end

const DINOV2 = (; patch_size = 14, init_values = 1e-5, img_size = 518, num_classes = 0)
const REG4 = (; reg_tokens = 4, no_embed_class = true)

for (name, args) in [
    "vit_small_patch14_dinov2" => (; DINOV2..., embed_dim = 384, depth = 12, num_heads = 6),
    "vit_base_patch14_dinov2" => (; DINOV2..., embed_dim = 768, depth = 12, num_heads = 12),
    "vit_large_patch14_dinov2" => (; DINOV2..., embed_dim = 1024, depth = 24, num_heads = 16),
    "vit_small_patch14_reg4_dinov2" =>
        (; DINOV2..., REG4..., embed_dim = 384, depth = 12, num_heads = 6),
    "vit_base_patch14_reg4_dinov2" =>
        (; DINOV2..., REG4..., embed_dim = 768, depth = 12, num_heads = 12),
    "vit_large_patch14_reg4_dinov2" =>
        (; DINOV2..., REG4..., embed_dim = 1024, depth = 24, num_heads = 16)
]
    register!(ArchSpec(
        TIMM, name, vit; family = "vision_transformer", defaults = args,
        references = get(VIT_REFERENCES, name, [])))
end

const CLIP = (; pre_norm = true, norm_eps = 1e-5)
const QUICKGELU = (; CLIP..., act_layer = "quick_gelu")
const CLIP_B = (; embed_dim = 768, depth = 12, num_heads = 12)
const CLIP_L = (; patch_size = 14, embed_dim = 1024, depth = 24, num_heads = 16)
const CLIP_H = (; patch_size = 14, embed_dim = 1280, depth = 32, num_heads = 16)

for (name, args) in [
    "vit_base_patch32_clip_224" => (; CLIP..., CLIP_B..., patch_size = 32),
    "vit_base_patch32_clip_256" =>
        (; CLIP..., CLIP_B..., patch_size = 32, img_size = 256, num_classes = 512),
    "vit_base_patch32_clip_384" => (; CLIP..., CLIP_B..., patch_size = 32, img_size = 384),
    "vit_base_patch16_clip_224" => (; CLIP..., CLIP_B..., patch_size = 16),
    "vit_base_patch16_clip_384" => (; CLIP..., CLIP_B..., patch_size = 16, img_size = 384),
    "vit_large_patch14_clip_224" => (; CLIP..., CLIP_L...),
    "vit_large_patch14_clip_336" => (; CLIP..., CLIP_L..., img_size = 336),
    "vit_huge_patch14_clip_224" => (; CLIP..., CLIP_H...),
    "vit_huge_patch14_clip_336" => (; CLIP..., CLIP_H..., img_size = 336),
    "vit_base_patch32_clip_quickgelu_224" =>
        (; QUICKGELU..., CLIP_B..., patch_size = 32, num_classes = 512),
    "vit_base_patch16_clip_quickgelu_224" =>
        (; QUICKGELU..., CLIP_B..., patch_size = 16, num_classes = 512),
    "vit_large_patch14_clip_quickgelu_224" => (; QUICKGELU..., CLIP_L..., num_classes = 768),
    "vit_large_patch14_clip_quickgelu_336" =>
        (; QUICKGELU..., CLIP_L..., img_size = 336, num_classes = 768),
    "vit_huge_patch14_clip_quickgelu_224" => (; QUICKGELU..., CLIP_H..., num_classes = 1024)
]
    register!(ArchSpec(
        TIMM, name, vit; family = "vision_transformer", defaults = args,
        references = get(VIT_REFERENCES, name, [])))
end

const SIGLIP = (; class_token = false, global_pool = "map", num_classes = 0)
const SIGLIP_GAP = (; class_token = false, global_pool = "avg", fc_norm = false,
    num_classes = 0)
const SIGLIP_B = (; embed_dim = 768, depth = 12, num_heads = 12)
const SIGLIP_L = (; embed_dim = 1024, depth = 24, num_heads = 16)
const SIGLIP_SO400M = (; embed_dim = 1152, depth = 27, num_heads = 16, mlp_ratio = 3.7362)

for (name, args) in [
    "vit_base_patch16_siglip_224" => (; SIGLIP..., SIGLIP_B..., patch_size = 16),
    "vit_base_patch16_siglip_256" =>
        (; SIGLIP..., SIGLIP_B..., patch_size = 16, img_size = 256),
    "vit_base_patch16_siglip_384" =>
        (; SIGLIP..., SIGLIP_B..., patch_size = 16, img_size = 384),
    "vit_base_patch16_siglip_512" =>
        (; SIGLIP..., SIGLIP_B..., patch_size = 16, img_size = 512),
    "vit_base_patch32_siglip_256" => (; SIGLIP..., SIGLIP_B..., patch_size = 32,
        img_size = 256, act_layer = "gelu_tanh"),
    "vit_large_patch16_siglip_256" =>
        (; SIGLIP..., SIGLIP_L..., patch_size = 16, img_size = 256),
    "vit_large_patch16_siglip_384" =>
        (; SIGLIP..., SIGLIP_L..., patch_size = 16, img_size = 384),
    "vit_large_patch16_siglip_512" => (; SIGLIP..., SIGLIP_L..., patch_size = 16,
        img_size = 512, act_layer = "gelu_tanh"),
    "vit_so400m_patch14_siglip_224" => (; SIGLIP..., SIGLIP_SO400M..., patch_size = 14),
    "vit_so400m_patch14_siglip_384" =>
        (; SIGLIP..., SIGLIP_SO400M..., patch_size = 14, img_size = 384),
    "vit_so400m_patch16_siglip_256" => (; SIGLIP..., SIGLIP_SO400M..., patch_size = 16,
        img_size = 256, act_layer = "gelu_tanh"),
    "vit_so400m_patch16_siglip_384" => (; SIGLIP..., SIGLIP_SO400M..., patch_size = 16,
        img_size = 384, act_layer = "gelu_tanh"),
    "vit_base_patch16_siglip_gap_224" => (; SIGLIP_GAP..., SIGLIP_B..., patch_size = 16),
    "vit_large_patch16_siglip_gap_256" =>
        (; SIGLIP_GAP..., SIGLIP_L..., patch_size = 16, img_size = 256),
    "vit_so400m_patch14_siglip_gap_224" =>
        (; SIGLIP_GAP..., SIGLIP_SO400M..., patch_size = 14)
]
    register!(ArchSpec(
        TIMM, name, vit; family = "vision_transformer", defaults = args,
        references = get(VIT_REFERENCES, name, [])))
end
