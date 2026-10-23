# The config.json of a timm repository: `architecture`, `model_args`, `num_classes` and a
# `pretrained_cfg`, which holds the preprocessing and the name of the classifier.
struct TimmFormat <: ConfigFormat end
const TIMM = TimmFormat()
Base.show(io::IO, ::TimmFormat) = print(io, "timm")

architecture_of(::TimmFormat, config) = get(config, "architecture", nothing)

function model_args(::TimmFormat, spec, config, kwargs)
    ckpt_classes = get(config, "num_classes", nothing)
    from_config = ckpt_classes === nothing ? (;) : (; num_classes = ckpt_classes)
    cfg = get(config, "pretrained_cfg", Dict())
    # timm's _update_default_model_kwargs: models of a fixed input size are built for it
    from_cfg = get(cfg, "fixed_input_size", false) && haskey(cfg, "input_size") ?
               (; img_size = Tuple(cfg["input_size"][2:3])) : (;)
    from_args = nt(get(config, "model_args", Dict()))
    return merge(spec.defaults, from_cfg, from_args, from_config, kwargs)
end

function skip_prefix(::TimmFormat, config, args)
    cfg = get(config, "pretrained_cfg", Dict())
    if haskey(args, :in_chans) && args.in_chans != get(get(cfg, "input_size", [3]), 1, 3)
        error("loading pretrained weights with a different `in_chans` is not supported yet")
    end
    ckpt_classes = get(config, "num_classes", nothing)
    (ckpt_classes === nothing || args.num_classes == ckpt_classes) && return nothing
    classifier = get(cfg, "classifier", nothing)
    classifier === nothing &&
        error("config.json does not name the classifier; cannot change `num_classes`")
    return classifier
end

"""
    Metalhead.Timm.pretrained_cfg(name; revision = nothing) -> NamedTuple

Preprocessing settings of a timm checkpoint (`mean`, `std`, `input_size`, `crop_pct`,
`interpolation`, ...), as stored in its `config.json`. `name` is a Hub repository, a bare
architecture (for its default checkpoint) or a local directory, as for
[`create_model`](@ref). `input_size` is in PyTorch order (C, H, W).
"""
function pretrained_cfg(
        name::AbstractString; revision::Union{Nothing, AbstractString} = nothing)
    config = JSON.parsefile(source_file(model_source(name, revision), "config.json"))
    return nt(get(config, "pretrained_cfg", Dict()))
end
