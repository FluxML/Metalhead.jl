# Ported from timm.models.convmixer (timm 1.0.30, Apache-2.0, Ross Wightman).
# Validated against test/fixtures/convmixer* and test/references/timm--convmixer_*.

# timm's `Residual`.
@concrete struct ConvMixerResidual
    fn
end
Flux.@layer ConvMixerResidual

(m::ConvMixerResidual)(x) = m.fn(x) .+ x

@concrete struct ConvMixer
    stem
    blocks
    pooling
    head_drop
    head
end
Flux.@layer ConvMixer

forward_features(m::ConvMixer, x) = m.blocks(m.stem(x))

function forward_head(m::ConvMixer, x; pre_logits::Bool = false)
    x = m.head_drop(m.pooling(x))
    return pre_logits ? x : m.head(x)
end

(m::ConvMixer)(x) = forward_head(m, forward_features(m, x))

const CONVMIXER_ACTIVATIONS = Dict("gelu" => gelu_erf, "relu" => relu)

"""
    convmixer(; dim, depth, kernel_size = 9, patch_size = 7, num_classes = 1000, kwargs...)

Port of timm's `ConvMixer`. Keyword arguments are those of the timm class, with `act_layer`
given by name (`"gelu"` or `"relu"`); `kernel_size` and `patch_size` are integers or
`(height, width)` tuples.
"""
function convmixer(; dim, depth, kernel_size = 9, patch_size = 7, in_chans = 3,
        num_classes = 1000, global_pool = "avg", drop_rate = 0.0, act_layer = "gelu")
    haskey(CONVMIXER_ACTIVATIONS, act_layer) ||
        error("`act_layer = $(repr(act_layer))` is not supported by the Metalhead port " *
              "of ConvMixer, which has \"gelu\" and \"relu\"")
    act = CONVMIXER_ACTIVATIONS[act_layer]
    patch = reverse(to_2tuple(patch_size))
    stem = Chain(Conv(patch, in_chans => dim; stride = patch), act, BatchNorm(dim))
    blocks = [Chain(
                  ConvMixerResidual(Chain(
                      create_conv2d(dim, dim, kernel_size; padding = "same",
                          depthwise = true, bias = true),
                      act, BatchNorm(dim))),
                  Conv((1, 1), dim => dim), act, BatchNorm(dim)) for _ in 1:depth]
    pooling, head_drop, head = create_classifier(dim, num_classes; pool_type = global_pool,
        drop_rate)
    return ConvMixer(stem, Chain(blocks), pooling, head_drop, head)
end

const CONVMIXER_REFERENCES = Dict(
    "convmixer_768_32" =>
        ["timm/convmixer_768_32.in1k" => "09fa91bbf9d93bf517098727fdd0ff580e69e035"]
)

for (name, args) in [
    "convmixer_1536_20" => (; dim = 1536, depth = 20, kernel_size = 9, patch_size = 7),
    "convmixer_768_32" => (; dim = 768, depth = 32, kernel_size = 7, patch_size = 7,
        act_layer = "relu"),
    "convmixer_1024_20_ks9_p14" => (; dim = 1024, depth = 20, kernel_size = 9,
        patch_size = 14)
]
    register!(ArchSpec(TIMM, name, convmixer; defaults = args,
        references = get(CONVMIXER_REFERENCES, name, [])))
end
