# Helpers of the tests in test/gpu/, which run only with METALHEAD_TEST_GPU=true.
module GPUSetup
using Metalhead
include(joinpath(pkgdir(Metalhead), "test", "parity_setup.jl"))
using .ParitySetup
using CUDA, cuDNN
using Flux: Flux, cpu, gpu_device

export GPU, GPU_RTOL, BF16_RTOL, fixtures_of, is_pipeline, gpu_inputs, check_gpu_fixture,
       gpu_cpu_error, check_gradient, check_half_precision, check_pipeline_fixture, fixture,
       fixture_model

CUDA.allowscalar(false)

# Fails without a functional GPU, rather than testing on the CPU.
const GPU = gpu_device(; force = true)

# By default (`CUDA.math_mode() == CUDA.DEFAULT_MATH`) cuDNN convolutions run in TF32, so the
# Float32 fixtures differ from their references by up to 3e-3 on the GPU, against 1e-6 on
# the CPU or with `CUDA.PEDANTIC_MATH`.
const GPU_RTOL = 1e-2
# In BFloat16, the tiny random models differ from Float32 by 0.6 to 3% (more for Qwen3.5 and
# the FLUX.2 transformer, as in PyTorch).
const BF16_RTOL = 5e-2

const FIXTURES = joinpath(pkgdir(Metalhead), "test", "fixtures")

is_pipeline(name) = isfile(joinpath(fixture(name), "model_index.json"))

"""
    fixtures_of(library) -> Vector{String}

The fixtures of `library` (`:Timm`, `:Transformers` or `:Diffusers`, the module of the
config format that `create_model` identifies), pipelines included.
"""
function fixtures_of(library::Symbol)
    return filter(readdir(FIXTURES)) do name
        config = is_pipeline(name) ? "model_index.json" : "config.json"
        format, _ = Metalhead.identify(JSON.parsefile(joinpath(fixture(name), config)))
        return nameof(parentmodule(typeof(format))) == library
    end
end

to_gpu(x::AbstractArray{<:AbstractFloat}) = GPU(Float32.(x))
to_gpu(x::AbstractArray) = GPU(x)
to_gpu(x) = x

fixture_acts(name) = Metalhead.Checkpoint(joinpath(fixture(name), "acts.safetensors"))

"""
    gpu_inputs(name) -> (args, kwargs)

The positional and keyword inputs of the fixture `name` on the GPU, in Float32.
"""
function gpu_inputs(name)
    args, kwargs = fixture_inputs(fixture_acts(name))
    return map(to_gpu, args), map(to_gpu, kwargs)
end

outputs(y) = y isa NamedTuple ? values(y) : (y,)

"""
    check_gpu_fixture(name) -> NamedTuple

Run the Float32 model of the fixture `name` on the GPU, on the fixture's inputs: the
relative errors of its output and features (`0.0` without features) against the fixture's,
and whether every output is a GPU array.
"""
function check_gpu_fixture(name)
    acts = fixture_acts(name)
    args, kwargs = gpu_inputs(name)
    model = GPU(fixture_model(fixture(name); eltype = Float32))
    y = model(args...; kwargs...)
    features = haskey(acts, "features") ?
               relerr(cpu(forward_features(model, args...; kwargs...)), acts["features"]) :
               0.0
    return (; output = Parity.output_error(cpu(y), acts), features,
        on_gpu = all(a -> a isa CuArray, outputs(y)))
end

"""
    gpu_cpu_error(name, imsize) -> Float64

The relative error of the Float32 model of the image fixture `name` on the GPU against the
CPU, on a random batch of 2 images of `imsize × imsize` pixels.
"""
function gpu_cpu_error(name, imsize)
    x = randn(Float32, imsize, imsize, size(fixture_acts(name)["input"], 3), 2)
    model = fixture_model(fixture(name); eltype = Float32)
    return relerr(cpu(GPU(model)(GPU(x))), model(x))
end
end
