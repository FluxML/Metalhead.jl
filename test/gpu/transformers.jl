using Metalhead, Test
include("../gpu_setup.jl")
using .GPUSetup
using Flux: BFloat16

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
