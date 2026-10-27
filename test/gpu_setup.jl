# Helpers of the tests in test/gpu/, which run only with METALHEAD_TEST_GPU=true.
module GPUSetup
using Metalhead
include(joinpath(pkgdir(Metalhead), "test", "parity_setup.jl"))
using .ParitySetup
using CUDA, cuDNN
using Flux: Flux, cpu, gpu_device

export GPU, GPU_RTOL, BF16_RTOL, BF16_CONV, fixtures_of, is_pipeline, gpu_inputs,
       check_gpu_fixture, check_gpu_st_fixture, check_gradient, check_half_precision,
       check_pipeline_fixture, fixture, fixture_model, IdsTokenizer

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
# cuDNN has no BFloat16 convolutions below compute capability 8.0 (the Turing GPUs of CI).
const BF16_CONV = CUDA.capability(CUDA.device()) >= v"8"

is_pipeline(name) = isfile(joinpath(fixture(name), "model_index.json"))

"""
    fixtures_of(library) -> Vector{String}

The fixtures of `library` (`:Timm`, `:Transformers`, `:Diffusers` or
`:SentenceTransformers`, the module of the config format that `create_model` identifies),
pipelines included; not those without a configuration file (test/fixtures/processors,
st_pooling).
"""
function fixtures_of(library::Symbol)
    return filter(test_data_names("fixtures")) do name
        dir = fixture(name)
        any(f -> isfile(joinpath(dir, f)), Metalhead.CONFIG_FILES) || return false
        config = Metalhead.source_config(Metalhead.model_source(dir, nothing))
        format, _ = Metalhead.identify(config)
        return nameof(parentmodule(typeof(format))) == library
    end
end

to_gpu(x::AbstractArray{<:AbstractFloat}) = GPU(Float32.(x))
to_gpu(x::AbstractVector{<:AbstractArray}) = map(to_gpu, x)
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

outputs(y) = y isa NamedTuple ? Tuple(Iterators.flatten(map(outputs, values(y)))) : (y,)

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
    check_gpu_st_fixture(name) -> Vector{NamedTuple}

Run the Float32 sentence-transformers model of the fixture `name` on the GPU, on the token
ids of each of its calls: the relative errors of the sentence embeddings, and of the token
embeddings of the texts' tokens, against the CPU's (which
test/models/sentence_transformers.jl compares with sentence-transformers'), and whether
both are GPU arrays.
"""
function check_gpu_st_fixture(name)
    acts = fixture_acts(name)
    model = create_model(fixture(name); eltype = Float32)
    gpu_model = GPU(model)
    calls = count(k -> occursin(r"^calls/\d+/input_ids$", k), keys(acts))
    return map(1:calls) do i
        key(k) = "calls/$i/$k"
        ids, mask = pad_right(acts[key("input_ids")] .+ 1, acts[key("attention_mask")])
        inputs = (; input_ids = ids, attention_mask = mask)
        haskey(acts, key("prompt_length")) &&
            (inputs = merge(inputs, (; prompt_length = acts[key("prompt_length")][])))
        y, g = model(inputs), gpu_model(map(to_gpu, inputs))
        tokens = Bool.(mask)
        return (;
            sentence_embedding = relerr(cpu(g.sentence_embedding), y.sentence_embedding),
            token_embeddings = relerr(cpu(g.token_embeddings)[:, tokens],
                y.token_embeddings[:, tokens]),
            on_gpu = g.sentence_embedding isa CuArray && g.token_embeddings isa CuArray)
    end
end

# The tokens of each column first: the tokenizer of a fixture may pad on the left, where
# Metalhead's models take padding on the right
function pad_right(ids, mask)
    order = [sortperm(mask[:, j]; rev = true) for j in axes(mask, 2)]
    reorder(x) = stack(x[o, j] for (j, o) in enumerate(order))
    return reorder(ids), reorder(mask)
end
end
