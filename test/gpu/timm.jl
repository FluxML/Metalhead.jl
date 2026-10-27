using Metalhead, Test
include("../gpu_setup.jl")
using .GPUSetup
using Flux: BFloat16

@testset "fixtures" begin
    @testset "$name" for name in fixtures_of(:Timm)
        r = check_gpu_fixture(name)
        @test r.on_gpu
        @test r.output <= GPU_RTOL
        @test r.features <= GPU_RTOL
    end
end

@testset "gradient" begin
    for name in ["mobilenetv4_conv_blur_medium", "vgg_adaptive_w", "gmlp", "res2net"]
        model = GPU(fixture_model(fixture(name); eltype = Float32))
        @test check_gradient(model, first(gpu_inputs(name))...)
    end
end

@testset "bfloat16" begin
    for name in ["resnet_basic", "vit_tiny"]
        model = GPU(fixture_model(fixture(name); eltype = Float32))
        @test check_half_precision(model, first(gpu_inputs(name))...; T = BFloat16,
            rtol = BF16_RTOL) skip = !BF16_CONV
    end
end
