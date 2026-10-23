# Ported from timm.models.mlp_mixer (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/{mixer,gmixer,resmlp,gmlp} and
# test/references/timm--{mixer,resmlp,gmlp}_*. `token_drop_path` is defined in vit.jl.

using Statistics: mean

# Tokens are (C, L, B), torch's (B, L, C) reversed. Token mixing acts on the token axis.
transpose_tokens(x) = permutedims(x, (2, 1, 3))

# timm.layers.lecun_normal_: truncated normal with std sqrt(1 / fan_in) / .8796.
function lecun_normal(dims...)
    std = sqrt(1 / first(Flux.nfan(dims...))) / 0.87962566103423978
    return Flux.truncated_normal(dims...; std, lo = -2std, hi = 2std)
end

@concrete struct MixerBlock
    norm1
    mlp_tokens
    drop_path
    norm2
    mlp_channels
end
Flux.@layer MixerBlock

function (m::MixerBlock)(x)
    x = x .+ m.drop_path(transpose_tokens(m.mlp_tokens(transpose_tokens(m.norm1(x)))))
    return x .+ m.drop_path(m.mlp_channels(m.norm2(x)))
end

# timm's `Affine`, the ResMLP norm: `beta + alpha * x` per channel.
@concrete struct ResMLPAffine
    alpha
    beta
end
Flux.@layer ResMLPAffine

function ResMLPAffine(dim::Integer)
    return ResMLPAffine(ones(Float32, dim, 1, 1), zeros(Float32, dim, 1, 1))
end

(m::ResMLPAffine)(x) = m.beta .+ m.alpha .* x

# timm's `ResBlock`: ResMLP block with a linear token mixer and layer scale.
@concrete struct ResMLPBlock
    norm1
    linear_tokens
    drop_path
    norm2
    mlp_channels
    ls1
    ls2
end
Flux.@layer ResMLPBlock

function (m::ResMLPBlock)(x)
    y = transpose_tokens(m.linear_tokens(transpose_tokens(m.norm1(x))))
    x = x .+ m.drop_path(m.ls1 .* y)
    return x .+ m.drop_path(m.ls2 .* m.mlp_channels(m.norm2(x)))
end

@concrete struct SpatialGatingUnit
    norm
    proj
end
Flux.@layer SpatialGatingUnit

# Not a `SpatialGatingUnit` method: it would replace the two-field constructor.
function make_spatial_gating_unit(dim, seq_len)
    init(dims...) = 1.0f-6 .* randn(Float32, dims...)
    proj = Dense(seq_len => seq_len; init, bias = ones(Float32, seq_len))
    return SpatialGatingUnit(LayerNorm(dim ÷ 2; eps = 1e-5), proj)
end

function (m::SpatialGatingUnit)(x)
    u, v = Flux.chunk(x, 2; dims = 1)
    return u .* transpose_tokens(m.proj(transpose_tokens(m.norm(v))))
end

@concrete struct SpatialGatingBlock
    norm
    mlp_channels
    drop_path
end
Flux.@layer SpatialGatingBlock

function (m::SpatialGatingBlock)(x)
    return x .+ m.drop_path(m.mlp_channels(m.norm(x)))
end

# timm's init_weights of GluMlp: the gate half of fc1 starts near zero with bias one.
function init_glu_gate!(m::GluMlp)
    h = size(m.fc1.weight, 1) ÷ 2
    m.fc1.weight[(h + 1):end, :] .= 1.0f-6 .* randn(Float32, h, size(m.fc1.weight, 2))
    m.fc1.bias[(h + 1):end] .= 1
    return m
end

const MIXER_ACTIVATIONS = Dict("gelu" => gelu_erf, "silu" => swish)
const MIXER_NORMS = Dict(
    "layernorm" => dim -> LayerNorm(dim; eps = 1e-6), "affine" => ResMLPAffine)
# The (block, MLP) pairs the fixtures cover.
const MIXER_BLOCK_MLPS = Dict(
    "MixerBlock" => ("Mlp", "GluMlp"), "ResBlock" => ("Mlp",),
    "SpatialGatingBlock" => ("GatedMlp",))

function check_choice(option, value, choices)
    value in choices && return nothing
    options = join(map(repr, sort!(collect(choices))), ", ")
    error("`$option = $(repr(value))` is not supported by the Metalhead port of " *
          "MlpMixer, which has $options")
end

function mixer_mlp(name, in_features, hidden_features; act_layer, drop,
        gate_layer = nothing)
    init = Flux.glorot_uniform
    name == "Mlp" && return Mlp(in_features, hidden_features; act_layer, drop, init)
    name == "GatedMlp" &&
        return GatedMlp(in_features, hidden_features; act_layer, drop, init, gate_layer)
    return init_glu_gate!(GluMlp(in_features, hidden_features; act_layer, drop, init))
end

function make_mixer_block(dim, seq_len, mlp_ratio; mlp_layer, norm_layer, act_layer, drop,
        drop_path)
    ratios = mlp_ratio isa Real ? (mlp_ratio, mlp_ratio) : Tuple(mlp_ratio)
    tokens_dim, channels_dim = map(r -> trunc(Int, r * dim), ratios)
    return MixerBlock(
        norm_layer(dim), mixer_mlp(mlp_layer, seq_len, tokens_dim; act_layer, drop),
        token_drop_path(drop_path), norm_layer(dim),
        mixer_mlp(mlp_layer, dim, channels_dim; act_layer, drop))
end

function make_res_mlp_block(dim, seq_len, mlp_ratio; mlp_layer, norm_layer, act_layer,
        init_values, drop, drop_path)
    mlp_ratio isa Real || error("ResBlock needs a single `mlp_ratio`, got $mlp_ratio")
    channels_dim = trunc(Int, dim * mlp_ratio)
    linear_tokens = Dense(seq_len => seq_len; init = Flux.glorot_uniform)
    return ResMLPBlock(norm_layer(dim), linear_tokens, token_drop_path(drop_path),
        norm_layer(dim), mixer_mlp(mlp_layer, dim, channels_dim; act_layer, drop),
        fill(Float32(init_values), dim), fill(Float32(init_values), dim))
end

function make_spatial_gating_block(dim, seq_len, mlp_ratio; mlp_layer, norm_layer,
        act_layer, drop, drop_path)
    mlp_ratio isa Real ||
        error("SpatialGatingBlock needs a single `mlp_ratio`, got $mlp_ratio")
    gate_layer = hidden -> make_spatial_gating_unit(hidden, seq_len)
    channels_dim = trunc(Int, dim * mlp_ratio)
    mlp = mixer_mlp(mlp_layer, dim, channels_dim; act_layer, drop, gate_layer)
    return SpatialGatingBlock(norm_layer(dim), mlp, token_drop_path(drop_path))
end

@concrete struct MlpMixer
    stem
    blocks
    norm
    global_pool
    head_drop
    head
end
Flux.@layer MlpMixer

forward_features(m::MlpMixer, x) = m.norm(m.blocks(m.stem(x)))

function forward_head(m::MlpMixer, x; pre_logits::Bool = false)
    m.global_pool == "avg" && (x = dropdims(mean(x; dims = 2); dims = 2))
    x = m.head_drop(x)
    return pre_logits ? x : m.head(x)
end

(m::MlpMixer)(x) = forward_head(m, forward_features(m, x))

"""
    mlp_mixer(; img_size = 224, patch_size = 16, num_blocks = 8, embed_dim = 512, kwargs...)

Port of timm's `MlpMixer`: MLP-Mixer, ResMLP, gMLP and the GLU mixer. Keyword arguments are
those of the timm class, with the classes it takes given by name: `block_layer` is
`"MixerBlock"`, `"ResBlock"` or `"SpatialGatingBlock"`, `mlp_layer` is `"Mlp"`, `"GluMlp"`
or `"GatedMlp"` (only the pairs of timm's entry points), `norm_layer` is `"layernorm"`
(`eps = 1e-6`) or `"affine"` and `act_layer` is `"gelu"` or `"silu"`. `init_values` is the
layer scale of `ResBlock` (`functools.partial(ResBlock, init_values = ...)` in timm).
`global_pool` is `"avg"` or `""`, which applies the head to every token.
"""
function mlp_mixer(; num_classes = 1000, img_size = 224, in_chans = 3, patch_size = 16,
        num_blocks = 8, embed_dim = 512, mlp_ratio = (0.5, 4.0), block_layer = "MixerBlock",
        mlp_layer = "Mlp", norm_layer = "layernorm", act_layer = "gelu", init_values = 1e-4,
        drop_rate = 0.0, proj_drop_rate = 0.0, drop_path_rate = 0.0, nlhb = false,
        stem_norm = false, global_pool = "avg")
    check_unsupported("MlpMixer"; stem_norm = (stem_norm, false))
    check_choice("block_layer", block_layer, keys(MIXER_BLOCK_MLPS))
    check_choice("mlp_layer", mlp_layer, MIXER_BLOCK_MLPS[block_layer])
    check_choice("norm_layer", norm_layer, keys(MIXER_NORMS))
    check_choice("act_layer", act_layer, keys(MIXER_ACTIVATIONS))
    check_choice("global_pool", global_pool, ("avg", ""))
    ph, pw = to_2tuple(patch_size)
    ih, iw = to_2tuple(img_size)
    seq_len = (ih ÷ ph) * (iw ÷ pw)
    norm = MIXER_NORMS[norm_layer]
    make_block = block_layer == "MixerBlock" ? make_mixer_block :
                 block_layer == "SpatialGatingBlock" ? make_spatial_gating_block :
                 (args...; kwargs...) -> make_res_mlp_block(args...; kwargs..., init_values)
    blocks = [make_block(embed_dim, seq_len, mlp_ratio; mlp_layer, norm_layer = norm,
                  act_layer = MIXER_ACTIVATIONS[act_layer], drop = proj_drop_rate,
                  drop_path = drop_path_rate)
              for _ in 1:num_blocks]
    stem = PatchEmbed(Conv((pw, ph), in_chans => embed_dim; stride = (pw, ph),
        init = lecun_normal))
    head = identity
    if num_classes > 0
        head_bias = nlhb ? fill(Float32(-log(num_classes)), num_classes) : true
        head = Dense(embed_dim => num_classes; init = Flux.zeros32, bias = head_bias)
    end
    return MlpMixer(stem, Chain(blocks), norm(embed_dim), global_pool,
        dropout_or_identity(drop_rate), head)
end

const MIXER_REFERENCES = Dict(
    "mixer_b16_224" =>
        ["timm/mixer_b16_224.goog_in21k_ft_in1k" =>
         "0006300d604941f96f16cb7d229137363e1b2ac6"],
    "resmlp_12_224" =>
        ["timm/resmlp_12_224.fb_in1k" => "59a4112ca93db165fd828bef7d47d758ad84295b"],
    "gmlp_s16_224" =>
        ["timm/gmlp_s16_224.ra3_in1k" => "ef4294efcbaed5fb374565716d9e08e4ed69a6d8"]
)

let mixer = (; patch_size = 16),
    resmlp = (; patch_size = 16, embed_dim = 384, mlp_ratio = 4, block_layer = "ResBlock",
        norm_layer = "affine"),
    gmlp = (; patch_size = 16, num_blocks = 30, block_layer = "SpatialGatingBlock",
        mlp_layer = "GatedMlp"),
    gmixer = (; patch_size = 16, embed_dim = 384, mlp_ratio = (1.0, 4.0),
        mlp_layer = "GluMlp", act_layer = "silu")

    for (name, args) in [
        "mixer_s32_224" => (; num_blocks = 8, embed_dim = 512, patch_size = 32),
        "mixer_s16_224" => (; num_blocks = 8, embed_dim = 512, mixer...),
        "mixer_b32_224" => (; num_blocks = 12, embed_dim = 768, patch_size = 32),
        "mixer_b16_224" => (; num_blocks = 12, embed_dim = 768, mixer...),
        "mixer_l32_224" => (; num_blocks = 24, embed_dim = 1024, patch_size = 32),
        "mixer_l16_224" => (; num_blocks = 24, embed_dim = 1024, mixer...),
        "gmixer_12_224" => (; num_blocks = 12, gmixer...),
        "gmixer_24_224" => (; num_blocks = 24, gmixer...),
        "resmlp_12_224" => (; resmlp..., num_blocks = 12),
        "resmlp_24_224" => (; resmlp..., num_blocks = 24, init_values = 1e-5),
        "resmlp_36_224" => (; resmlp..., num_blocks = 36, init_values = 1e-6),
        "resmlp_big_24_224" => (; resmlp..., num_blocks = 24, embed_dim = 768,
            patch_size = 8, init_values = 1e-6),
        "gmlp_ti16_224" => (; gmlp..., embed_dim = 128, mlp_ratio = 6),
        "gmlp_s16_224" => (; gmlp..., embed_dim = 256, mlp_ratio = 6),
        "gmlp_b16_224" => (; gmlp..., embed_dim = 512, mlp_ratio = 6)
    ]
        register!(ArchSpec(:timm, name, mlp_mixer; defaults = args,
            references = get(MIXER_REFERENCES, name, [])))
    end
end
