using Metalhead, Test
include("../gpu_setup.jl")
using .GPUSetup
using Flux: BFloat16
using CUDA: CUDA

@testset "fixtures" begin
    @testset "$name" for name in fixtures_of(:Transformers)
        r = check_gpu_fixture(name)
        @test r.on_gpu
        @test r.output <= GPU_RTOL
        @test r.features <= GPU_RTOL
    end
end

@testset "gradient" begin
    for name in ["llama_tiny", "granitemoe_tiny"]
        args, kwargs = gpu_inputs(name)
        model = GPU(fixture_model(fixture(name); eltype = Float32))
        @test check_gradient(model, args...; kwargs...)
    end
end

@testset "bfloat16" begin
    for name in ["llama_tiny", "granitemoe_tiny"]
        args, kwargs = gpu_inputs(name)
        model = GPU(fixture_model(fixture(name); eltype = Float32))
        @test check_half_precision(
            model, args...; T = BFloat16, rtol = BF16_RTOL, kwargs...)
    end
end

# A language model on the GPU takes a tokenizer's output padded on either side.
@testset "tokenizer outputs" begin
    model = fixture_model(fixture("llama_tiny"); eltype = Float32)
    ids = rand(1:64, 5, 2)
    mask = [1 0; 1 0; 1 1; 1 1; 1 1]   # the second padded on the left
    y = model((; input_ids = ids, attention_mask = mask))
    y_gpu = GPU(model)(GPU((; input_ids = ids, attention_mask = mask)))
    @test y_gpu isa CUDA.CuArray
    @test Array(y_gpu) ≈ y rtol = GPU_RTOL
end

