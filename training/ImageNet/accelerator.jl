using XLA, TensorFlow, Zygote

# By default, we use "CPU" versions that do nothing/execute directly using Flux/Julia
model_to_device = (model) -> model
model_to_host = (model) -> model
process_minibatch = (model, opt, x, y) -> begin
    y_hat = model(model_to_device(x))
    loss = Flux.crossentropy(model_to_device(y), y_hat)
    Flux.Optimise.@interrupts Flux.back!(loss)
    opt()
    return Flux.Tracker.data(model_to_host(loss))
end

# If we're using GPU devices, just use Flux.gpu() to copy tensors around,
# but otherwise do the exact same thing:
if !isempty(gpu_devices)
    model_to_device = (model) -> Flux.gpu(model)
    model_to_host = (model) -> Flux.cpu(model)
end



# If we're using XLA, it's whole 'nother ball game.  We first compile 
# two versions of our code (one that does just the forward pass and one
# that does the whole minibatch update)

# Compiled XLA Model struct
struct CompiledXLAModel
    ic_model::ImmutableChain
    allocation

    # Dicts containing compiled models for forward pass and minibatch updates,
    # keyed off of the input size.  These are dynamically created within the
    # minibatch and forward calls.
    forward_compilecache::Dict
    minibatch_compilecache::Dict
    
    sess
    device
end

# If we're using an XRT backend, load up the XLA package
if xrt_address !== nothing
    # Open our XRT session
    sess = Session(Graph(); target="grpc://$(xrt_address)")

    function get_device(sess, device_name)
        devices = collect(TensorFlow.DeviceList(sess))
        filt_devices = filter(x -> occursin(device_name, string(x)), devices)
        if isempty(filt_devices)
            error("Unable to find XLA device matching $(device_name)!  Devices: $(devices)")
        end
        device = first(filt_devices)

        inc_number(x::Number) = x + 1
        inc_number(x) = x
        function fixup_device(d::TensorFlow.Device)
            d.parts[:] .= [TensorFlow.DevicePart(p.kind, inc_number(p.index)) for p in d.parts]
            return d
        end
        fixup_device(d) = d
        return fixup_device(TensorFlow.Device(device.name))
    end
    # Convert string to Device Name
    xla_device = get_device(sess, xla_device)

    # If we're running on a TPU, configure it for distributed operation
    if occursin("TPU", string(xla_device))
        as_default(sess.graph) do
            with_device(xla_device) do
                run(sess, TensorFlow.Ops.configure_distributed_tpu())
            end
        end
    end


    # Compile forward pass on demand
    function compile_forward_for_device(model, x)
        if !haskey(model.forward_compilecache, size(x))
            TensorFlow.as_default(model.sess.graph) do
                with_device(model.device) do
                    # Compile the model forward pass
                    model.forward_compilecache[size(x)] = @tpu_compile model.ic_model(x)
                end
            end
        end
        return model.forward_compilecache[size(x)]
    end

    # Compile full minibatch update on demand
    function zygote_minibatch(model, opt, x, y)
        # Use Zygote to calculate both the forward pass and the derivative program
        y_hat, back = Zygote._forward(Zygote.Context{Nothing}(nothing), model.ic_model, model_to_device(x))
        loss = Flux.crossentropy(model_to_device(y), y_hat)
        
        # Perform magic here to apply `loss` to Param objects within model and update the parameters with `opt()`
        
        return Flux.Tracker.data(model_to_host(loss))
    end
    
    function compile_minibatch_for_device(model, opt, x, y)
        if !haskey(model.forward_compilecache, size(x))
            TensorFlow.as_default(model.sess.graph) do
                with_device(model.device) do
                    # Compile the model minibatch process
                    model.minibatch_compilecache[size(x)] = @tpu_compile zygote_minibatch(model, opt, x, y)
                end
            end
        end
        return model.minibatch_compilecache[size(x)]
    end

    # If we are given an XRTArray, grab the handle and turn it into
    # an XRTAllocation.  Otherwise, keep it as-is.
    get_xrthandle(x) = x
    get_xrthandle(x::XRTArray) = XLA.gethandle!(m.sess, x)

    # Directly calling the model subs out to the compiled forward batch,
    # as would be expected.
    function (m::CompiledXLAModel)(x)
        compiled_forward_code = compile_forward_for_device(model, x)
        return run(compiled_forward_code, m.allocation, get_xrthandle(x))
    end
    
    # Gotta Go Fast (TM)
    process_minibatch = (model, opt, x, y) -> begin
        compiled_minibatch_code = compile_minibatch_for_device(model)
        return run(compiled_minibatch_code, model.allocation, model, opt, get_xrthandle(x), get_xrthandle(y))
    end

    # To move a model to an XLA device, we move all constituent parameters
    # within the model, then we compile it, returning a `CompiledXLAModel`
    function xla(model::Union{Chain,Metalhead.ClassificationModel})
        ic_model = Flux.mapleaves(
            xla, ImmutableChain(model.layers...)
        )
        return TensorFlow.as_default(sess.graph) do
            return with_device(xla_device) do
                alloc = XRTAllocation(sess, XLA.struct_to_literal(model))
                return CompiledXLAModel(
                    alloc,
                    Dict(),
                    Dict(),
                    sess,
                    device
                )
            end
        end
    end

    # We assume that if someone calls `model_to_device()` on something we
    # don't recognize, we should just leave it alone.  But if it's an AA,
    # then move it over a la XRTArray!
    xla(x) = x
    xla(x::AbstractArray) = XRTArray(sess, x)

    # And finally, set that to be the definition of `model_to_device()`
    model_to_device = xla
    model_to_host = (x) -> convert(x, Array)
end
