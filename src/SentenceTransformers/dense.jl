# Ported from sentence_transformers.base.modules.dense (sentence-transformers 6.2.0,
# Apache-2.0, UKP Lab and the Hugging Face Team). Validated against
# test/fixtures/st_qwen3_dense.

"""
    Dense(in => out; bias = true, activation_function = tanh,
          module_input_name = "sentence_embedding", module_output_name = module_input_name,
          use_residual = false)

sentence-transformers' `Dense` module: the feature `module_input_name` of a NamedTuple of
features, through the Linear layer `linear` and `activation_function`, as the feature
`module_output_name`. With `use_residual`, the input is added to the output, through the
Linear layer `residual` (without bias) where `in != out`. Its weights are those of the
`model.safetensors` of its subfolder.
"""
@concrete struct Dense
    linear
    activation_function
    residual
    use_residual
    module_input_name
    module_output_name
end
Flux.@layer Dense trainable=(linear, residual)

function Dense((in, out)::Pair{<:Integer, <:Integer}; bias::Bool = true,
        activation_function = tanh, module_input_name = "sentence_embedding",
        module_output_name = nothing, use_residual::Bool = false)
    residual = use_residual && in != out ? Flux.Dense(in => out; bias = false) : nothing
    return Dense(Flux.Dense(in => out; bias), activation_function, residual, use_residual,
        String(module_input_name), String(something(module_output_name, module_input_name)))
end

function (m::Dense)(features::NamedTuple)
    x = features[Symbol(m.module_input_name)]
    y = m.activation_function.(m.linear(x))
    if m.use_residual
        y = y .+ (m.residual === nothing ? x : m.residual(x))
    end
    return merge(features, NamedTuple{(Symbol(m.module_output_name),)}((y,)))
end

# The activation functions of torch, by class name, as sentence-transformers saves them
# (`torch.nn.modules.activation.Tanh`)
const ACTIVATIONS = Dict("Tanh" => tanh, "Identity" => identity, "ReLU" => relu,
    "GELU" => gelu_erf, "Sigmoid" => sigmoid)

function activation(name, file)
    parts = split(name, '.')
    class = String(last(parts))
    first(parts) == "torch" && haskey(ACTIVATIONS, class) ||
        unsupported("activation_function", name, file,
            join(("torch.nn.$a" for a in sort!(collect(keys(ACTIVATIONS)))), ", "))
    return ACTIVATIONS[class]
end

function load_dense(source; pretrained, eltype)
    file = "config.json"
    dict = read_file(source, file)
    dict === nothing &&
        error("the Dense module of $(source_name(source)) has no config.json")
    c = read_settings(Dense, dict, file;
        defaults = (; in_features = nothing, out_features = nothing, bias = true,
            activation_function = "torch.nn.modules.activation.Tanh",
            module_input_name = "sentence_embedding", module_output_name = nothing,
            use_residual = false))
    c.in_features isa Integer && c.out_features isa Integer ||
        error("the Dense module of $(source_name(source)) has no `in_features` or `out_features`")
    m = Dense(c.in_features => c.out_features; c.bias, c.module_input_name,
        c.module_output_name, c.use_residual,
        activation_function = activation(c.activation_function, file))
    m = ELTYPE_CONVERTERS[eltype](m)
    if pretrained
        found = source_file_if_exists(source, "model.safetensors")
        found === nothing &&
            error("the Dense module of $(source_name(source)) has no model.safetensors; Metalhead loads safetensors weights only, not pytorch_model.bin")
        load_state!(m, Checkpoint(found))
    end
    return m
end
MODULE_LOADERS["Dense"] = load_dense
