# The config.json of a diffusers model: the class in `_class_name` and the arguments of its
# constructor, among keys of diffusers' own (`_diffusers_version`, `_name_or_path`) that the
# models don't read. Its weights are `diffusion_pytorch_model.safetensors`.
struct DiffusersFormat <: AbstractConfigFormat end
const DIFFUSERS = DiffusersFormat()
Base.show(io::IO, ::DiffusersFormat) = print(io, "diffusers")
direct_weight_init(::DiffusersFormat) = true

# The model_index.json of a diffusers pipeline: its class in `_class_name`, and each component
# as `name => [library, class]`, saved in the subfolder `name` (`[null, null]` if absent).
struct DiffusersPipelineFormat <: AbstractConfigFormat end
const DIFFUSERS_PIPELINE = DiffusersPipelineFormat()
Base.show(io::IO, ::DiffusersPipelineFormat) = print(io, "diffusers pipeline")

is_component(v) = v isa AbstractVector && length(v) == 2 &&
                  all(x -> x === nothing || x isa AbstractString, v)
is_diffusers(config) = haskey(config, "_class_name") && haskey(config, "_diffusers_version")
# diffusers names every pipeline class `...Pipeline`
is_pipeline(config) = endswith(config["_class_name"], "Pipeline")

function architecture_of(::DiffusersFormat, config)
    is_diffusers(config) && !is_pipeline(config) || return nothing
    return config["_class_name"]
end

function architecture_of(::DiffusersPipelineFormat, config)
    is_diffusers(config) && is_pipeline(config) || return nothing
    return config["_class_name"]
end

weights_name(::DiffusersFormat) = "diffusion_pytorch_model"

# diffusers passes a model the keys of its config.json that its constructor declares.
function model_args(::Union{DiffusersFormat, DiffusersPipelineFormat}, spec, config, kwargs)
    return merge(spec.defaults, declared_args(spec.build, config), kwargs)
end

# Schedulers have no weights, and their own scheduler_config.json: they are built from it by
# class name, and are not architectures of the registry.
const SCHEDULERS = Dict{String, Any}()

# The components of a pipeline are built from their subfolders, with the same `pretrained`
# and `eltype`; the builder of the pipeline takes them as keyword arguments, with the other
# keys of model_index.json it declares. A tokenizer is HuggingFaceTokenizers' `Tokenizer`
# (see `pipeline_tokenizer`).
function instantiate(format::DiffusersPipelineFormat, spec, source, config;
        pretrained::Bool, eltype, kwargs...)
    components = map(collect(filter(p -> is_component(last(p)), config))) do (name, (library, class))
        sub = subsource(source, name)
        component = if class === nothing
            nothing
        elseif haskey(SCHEDULERS, class)
            build = SCHEDULERS[class]
            build(; declared_args(build, JSON.parsefile(source_file(sub, "scheduler_config.json")))...)
        elseif occursin("Tokenizer", class)
            pipeline_tokenizer(sub)
        else
            subconfig = source_config(sub)
            subformat, architecture = identify(subconfig)
            model = instantiate(subformat, lookup(architecture), sub, subconfig; pretrained,
                eltype)
            architecture == class ? model :
            base_model(model, class, architecture, "component `$name` of $(spec.name)")
        end
        return Symbol(name) => component
    end
    args = model_args(format, spec, filter(p -> !is_component(last(p)), config), NamedTuple(kwargs))
    return spec.build(; components..., args...)
end

# The tokenizer of a pipeline, from its subfolder: HuggingFaceTokenizers' `Tokenizer`, or
# without HuggingFaceTokenizers the `DeferredTokenizer` of its tokenizer.json, which the
# pipeline loads once HuggingFaceTokenizers is loaded; `nothing` without a tokenizer.json
# (a local pipeline saved without one).
pipeline_tokenizer(s::ModelSource) = load_tokenizer_or_defer(ProcessorFiles(s))

# A component that a pipeline holds as the base model of the architecture of its checkpoint,
# as transformers loads the decoder of a causal language model: Z-Image's `Qwen3Model` text
# encoder is saved as a `Qwen3ForCausalLM`.
function base_model(model, class, architecture, what)
    model isa CausalLM && class == replace(architecture, r"ForCausalLM$" => "Model") ||
        error("$what is a $class, but its config.json describes a $architecture")
    return first(decoder(model))
end

# The tokenizer with which a pipeline encodes prompts, from its `tokenizer`.
function prompt_tokenizer(tokenizer)
    tok = resolve_tokenizer(tokenizer)
    tok === nothing || return tok
    tokenizer isa DeferredTokenizer &&
        error("the tokenizer of the pipeline needs HuggingFaceTokenizers.jl to encode prompts: run `using HuggingFaceTokenizers`, and call the pipeline again, which needn't be loaded again; or pass prompt embeddings, or the token indices of the prompts to `encode_prompt`")
    return error("the pipeline has no tokenizer to encode prompts, as its repository has no tokenizer/tokenizer.json: pass prompt embeddings, or the token indices of the prompts to `encode_prompt`")
end

# Diffusers components have no processor of their own: a pipeline preprocesses in its code.
function instantiate_processor(::DiffusersFormat, files, config; kwargs...)
    return error("$(config["_class_name"]), a component of a diffusers pipeline, has no processor: the pipeline prepares its inputs (`create_processor(repo; subfolder = \"tokenizer\")` loads the tokenizer of a pipeline)")
end

function instantiate_processor(::DiffusersPipelineFormat, files, config; kwargs...)
    return error("$(config["_class_name"]) has no processor: it prepares its inputs itself, with the tokenizer `pipe.tokenizer` of `pipe = create_model(repo)`, which `create_processor(repo; subfolder = \"tokenizer\")` loads alone, as a `TextProcessor`")
end
