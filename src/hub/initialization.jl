# Pretrained construction allocates weights in their final precision, without running the
# initializers that the checkpoint replaces. Keep the context task-local: simultaneous
# create_model calls may request different types, and errors must restore normal construction.
weight_init_context() = get(task_local_storage(), :metalhead_weight_init, nothing)

function init_array(init, dims...; norm::Bool = false)
    context = weight_init_context()
    context === nothing && return init(dims...)
    T = norm ? promote_type(context.eltype, Float32) : context.eltype
    array = Array{T}(undef, dims...)
    context.initializers === nothing || (context.initializers[array] = init)
    return array
end

# Some timm families initialise their weights again after constructing the layers. Record
# that initializer too when a replacement classifier will need it, without filling weights
# that are about to load from the checkpoint.
function init_parameter!(array, init)
    context = weight_init_context()
    if context === nothing
        copyto!(array, init(size(array)...))
    elseif context.initializers !== nothing
        context.initializers[array] = init
    end
    return array
end

function init_layer(layer, args...; kwargs...)
    weight_init_context() === nothing && return layer(args...; kwargs...)
    return pretrained_layer(layer, args...; kwargs...)
end

function pretrained_layer(layer::Union{Type{Dense}, Type{Conv}, Type{ConvTranspose}},
        args...; init = Flux.glorot_uniform, kwargs...)
    return layer(args...; init = (dims...) -> init_array(init, dims...), kwargs...)
end

function pretrained_layer(::Type{Embedding}, args...; init = Flux.randn32, kwargs...)
    return Embedding(args...; init = (dims...) -> init_array(init, dims...), kwargs...)
end

# Flux's normalization layers have small, deterministic initializers. Their conversion
# methods also specify which parameters and statistics stay Float32 under mixed precision.
function pretrained_layer(layer::Union{Type{LayerNorm}, Type{BatchNorm}, Type{GroupNorm}},
        args...; kwargs...)
    return ELTYPE_CONVERTERS[weight_init_context().eltype](layer(args...; kwargs...))
end

function build_pretrained(build, args, eltype; skip = nothing)
    initializers = skip === nothing ? nothing : IdDict{Any, Any}()
    context = (; eltype, initializers)
    model = task_local_storage(:metalhead_weight_init, context) do
        return build(; args...)
    end
    if skip !== nothing
        for p in first(torch_params(model))
            (p.key == skip || startswith(p.key, skip * ".")) || continue
            haskey(initializers, p.array) || continue
            copyto!(p.array, initializers[p.array](size(p.array)...))
        end
    end
    return model
end
