# By default, do no acceleration
accelerator = identity

# If we're using GPU devices, import CuArrays and set our accelerator
if !isempty(gpu_devices)
    Core.eval(@__MODULE__, :(using CuArrays))
    Core.eval(Flux, :(gpu_adapter = $(CuArrays.cu)))

    accelerator = Flux.gpu
end


# Compiled XLA Model struct
struct CompiledXLAModel
    allocation
    compiled_code
    sess
    device
end

# If we're using an XRT backend, load up the XLA package
if xrt_address !== nothing
    Core.eval(@__MODULE__, :(using XLA, TensorFlow))

    # Open our XRT session
    sess = Session(Graph(); target="grpc://$(xrt_address)")

    function get_device(sess, device_name)
        devices = collect(TensorFlow.DeviceList(sess))
        filt_devices = filter(x -> occursin(device_name, x.name), devices)
        if isempty(filt_devices)
            error("Unable to find XLA device matching $(device_name)!  Devices: $(devices)")
        end

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
    if occursin("TPU", xla_device.name)
        as_default(sess.graph) do
            with_device(xla_device) do
                run(sess, TensorFlow.Ops.configure_distributed_tpu())
            end
        end
    end

    # Maps each array to an XRTArray, after combining it all together
    # into an ImmutableChain, so that we can more forcefully glom the whole
    # thing together into a single chunk of static code.

    # Compiles a model for a specific device and allocates space for it on that device.
    function compile_model_for_device(model::ImmutableChain, sess, device, batch_size)
        return TensorFlow.as_default(sess.graph) do
            return with_device(device) do
                alloc = XRTAllocation(sess, XLA.struct_to_literal(model))
                # Dummy input
                x_dummy = randn(Float32, 224, 224, 3, batch_size)
                compiled = @tpu_compile model(XRTArray(sess, x_dummy))
                return CompiledXLAModel(alloc, compiled, sess, device)
            end
        end
    end

    function (m::CompiledModel)(args...)
        # If we are given an XRTArray, grab the handle and turn it into
        # an XRTAllocation.  Otherwise, keep it as-is.
        transfer_array(x) = x
        transfer_array(x::XRTArray) = XLA.gethandle!(m.sess, x)
    
        # Gotta Go Fast (TM)
        return run(m.compiled_code, m.allocation, transfer_array.(args)...)
    end


    # "Accelerating" a model for XLA looks like compiling it, 
    function xla_accelerator(model::Chain)
        ic_model = Flux.mapleaves(
            x->isa(x, AbstractArray) ? XRTArray(x) : x,
            ImmutableChain(model.layers...)
        )
        return compile_model_for_device(
            ic_model,
            sess,
            xla_device,
            batch_size,
        )
    end

    # We assume that if someone calls `xla_accelerator()` on something we
    # don't recognize, then it's probably a VGG19 model, and so we sub out
    # to the embedded Chain object:
    xla_accelerator(x) = xla_accelerator(x.layers)

    # "Accelerating" an input batch is calling XRTArray() on it
    xla_accelerator(x::XRTAllocation) = x
    xla_accelerator(x::XRTArray) = x
    xla_accelerator(x::AbstractArray) = XRTArray(sess, x)

    # Set our accelerator() (which is used to both move input batches to our
    # accelerator device as well as move our models to the device)
    accelerator = xla_accelerator
end