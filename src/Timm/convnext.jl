# Ported from timm.models.convnext (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/convnext_* and test/references/timm--convnext*.

@concrete struct ConvNeXtBlock
    conv_dw
    norm
    mlp
    gamma
    shortcut
    drop_path
    use_conv_mlp
end
Flux.@layer ConvNeXtBlock

function ConvNeXtBlock(
        dim; kernel_size = 7, mlp_ratio = 4, conv_mlp = false, ls_init_value = 1e-6,
        drop_path_rate = 0.0)
    conv_dw = Conv((kernel_size, kernel_size), dim => dim;
        pad = get_padding(kernel_size), groups = dim,
        init = TRUNC_NORMAL)
    norm = conv_mlp ? LayerNorm2d(dim) : LayerNorm(dim; eps = 1e-6)
    mlp = Mlp(dim, floor(Int, mlp_ratio * dim); use_conv = conv_mlp)
    gamma = fill(Float32(ls_init_value), dim)
    return ConvNeXtBlock(conv_dw, norm, mlp, gamma, identity,
        something(drop_path(drop_path_rate), identity), conv_mlp)
end

function (b::ConvNeXtBlock)(x)
    shortcut = x
    x = b.conv_dw(x)
    if b.use_conv_mlp
        x = b.mlp(b.norm(x))
    else
        x = permutedims(x, (3, 1, 2, 4))  # WHCN to CWHN: channels first for LayerNorm and Dense
        x = b.mlp(b.norm(x))
        x = permutedims(x, (2, 3, 1, 4))
    end
    x = x .* reshape(b.gamma, 1, 1, :, 1)
    return b.drop_path(x) .+ b.shortcut(shortcut)
end

@concrete struct ConvNeXtStage
    downsample
    blocks
end
Flux.@layer ConvNeXtStage

# Not a `ConvNeXtStage` method: it would replace the two-field constructor.
function convnext_stage(
        in_chs, out_chs; kernel_size = 7, stride = 2, depth = 2, drop_path_rates = nothing,
        ls_init_value = 1e-6, conv_mlp = false)
    downsample = identity
    if in_chs != out_chs || stride > 1
        ks = stride > 1 ? 2 : 1
        downsample = Chain(LayerNorm2d(in_chs), Conv((ks, ks), in_chs => out_chs; stride, init = TRUNC_NORMAL))
    end
    rates = something(drop_path_rates, zeros(depth))
    blocks = [ConvNeXtBlock(
                  out_chs; kernel_size, conv_mlp, ls_init_value, drop_path_rate = rates[i])
              for i in 1:depth]
    return ConvNeXtStage(downsample, Chain(blocks))
end

(m::ConvNeXtStage)(x) = m.blocks(m.downsample(x))

@concrete struct ConvNeXt
    stem
    stages
    norm_pre
    head
end
Flux.@layer ConvNeXt

"""
    convnext(; depths, dims, num_classes = 1000, in_chans = 3, kwargs...)

Port of timm's `ConvNeXt` with the patch stem and the default head (ConvNeXt, ConvNeXt-Atto to Nano).
Keyword arguments are those of the timm class.
"""
function convnext(;
        in_chans = 3, num_classes = 1000, global_pool = "avg", output_stride = 32,
        depths = (3, 3, 9, 3), dims = (96, 192, 384, 768), kernel_sizes = 7,
        ls_init_value = 1e-6, stem_type = "patch", patch_size = 4, head_init_scale = 1.0,
        head_norm_first = false, head_hidden_size = nothing, conv_mlp = false,
        conv_bias = true, use_grn = false, act_layer = "gelu", norm_layer = nothing,
        norm_eps = nothing, drop_rate = 0.0, drop_path_rate = 0.0)
    check_unsupported(
        "ConvNeXt"; global_pool = (global_pool, "avg"), output_stride = (output_stride, 32),
        stem_type = (stem_type, "patch"), head_init_scale = (head_init_scale, 1.0),
        head_norm_first = (head_norm_first, false), head_hidden_size = (
            head_hidden_size, nothing),
        conv_bias = (conv_bias, true), use_grn = (use_grn, false), act_layer = (
            act_layer, "gelu"),
        norm_layer = (norm_layer, nothing), norm_eps = (norm_eps, nothing))
    ls_init_value === nothing &&
        error("`ls_init_value = nothing` is not supported by the Metalhead port of ConvNeXt")
    length(depths) == length(dims) == 4 || error("ConvNeXt ports have exactly 4 stages")
    kernel_sizes = kernel_sizes isa Integer ? ntuple(_ -> kernel_sizes, 4) :
                   Tuple(kernel_sizes)
    stem = Chain(
        Conv((patch_size, patch_size), in_chans => dims[1]; stride = patch_size, init = TRUNC_NORMAL),
        LayerNorm2d(dims[1]))
    dp_rates = collect(range(0, drop_path_rate; length = sum(depths)))
    offsets = cumsum([0; collect(depths[1:3])])
    curr_stride = patch_size
    stages = map(1:4) do i
        stride = curr_stride == 2 || i > 1 ? 2 : 1
        curr_stride >= output_stride && stride > 1 &&
            error("dilated stages are not supported by the Metalhead port of ConvNeXt")
        curr_stride *= stride
        return convnext_stage(
            i == 1 ? dims[1] : dims[i - 1], dims[i]; kernel_size = kernel_sizes[i],
            stride, depth = depths[i], ls_init_value, conv_mlp,
            drop_path_rates = dp_rates[(offsets[i] + 1):(offsets[i] + depths[i])])
    end
    return ConvNeXt(stem, Chain(stages...), identity, NormMlpClassifierHead(dims[end], num_classes; drop_rate))
end

forward_features(m::ConvNeXt, x) = m.norm_pre(m.stages(m.stem(x)))
forward_head(m::ConvNeXt, x; pre_logits::Bool = false) = m.head(x; pre_logits)
(m::ConvNeXt)(x) = forward_head(m, forward_features(m, x))

# The four stages; `norm` applies `norm_pre` to the last one.
function forward_intermediates(m::ConvNeXt, x; indices = nothing, norm::Bool = false,
        stop_early::Bool = false, output_fmt = "NCHW", intermediates_only::Bool = false)
    check_output_fmt(output_fmt)
    n = length(m.stages)
    take, max_index = feature_take_indices(n, indices)
    stop = stop_early ? max_index : n
    outs = stage_outputs(m.stages.layers, m.stem(x), stop)
    intermediates = [norm && i == n ? m.norm_pre(outs[i]) : outs[i] for i in take]
    intermediates_only && return intermediates
    return stop == n ? m.norm_pre(last(outs)) : last(outs), intermediates
end

function convnext_stride(s::ConvNeXtStage)
    return s.downsample === identity ? 1 : first(s.downsample[2].stride)
end

function feature_info(m::ConvNeXt)
    stages = m.stages.layers
    return [(; num_chs = length(last(s.blocks.layers).gamma),
                reduction = first(m.stem[1].stride) * prod(convnext_stride, stages[1:i]),
                module_name = "stages.$(i - 1)") for (i, s) in enumerate(stages)]
end

const CONVNEXT_REFERENCES = Dict(
    "convnext_atto" =>
        ["timm/convnext_atto.d2_in1k" => "435c0a8d02d5b872e90c8cde36887465982cb399"],
    "convnext_tiny" =>
        ["timm/convnext_tiny.fb_in1k" => "b43a6303c9fcf176d2d707478a128c2c91e93528"]
)

for (name, args) in [
    "convnext_atto" => (depths = (2, 2, 6, 2), dims = (40, 80, 160, 320), conv_mlp = true),
    "convnext_femto" => (depths = (2, 2, 6, 2), dims = (48, 96, 192, 384), conv_mlp = true),
    "convnext_pico" => (depths = (2, 2, 6, 2), dims = (64, 128, 256, 512), conv_mlp = true),
    "convnext_nano" => (depths = (2, 2, 8, 2), dims = (80, 160, 320, 640), conv_mlp = true),
    "convnext_tiny" => (depths = (3, 3, 9, 3), dims = (96, 192, 384, 768)),
    "convnext_small" => (depths = (3, 3, 27, 3), dims = (96, 192, 384, 768)),
    "convnext_base" => (depths = (3, 3, 27, 3), dims = (128, 256, 512, 1024)),
    "convnext_large" => (depths = (3, 3, 27, 3), dims = (192, 384, 768, 1536))
]
    register!(ArchSpec(TIMM, name, convnext; family = "convnext", defaults = args,
        references = get(CONVNEXT_REFERENCES, name, [])))
end
