using Metalhead, Test
include("../gpu_setup.jl")
using .GPUSetup
using Flux: BFloat16

# cuDNN's 3 × 3 mean pooling with padding 1 returns zeros or NaN on 1 × 1 and 2 × 2 maps,
# which these fixtures reach: they are checked against the CPU on larger images instead.
const MEANPOOL_IMSIZE = Dict("inception_v3" => 139, "inception_v4" => 139, "res2net" => 64,
    "res2next_deep" => 64)

@testset "fixtures" begin
    @testset "$name" for name in fixtures_of(:Timm)
        r = check_gpu_fixture(name)
        @test r.on_gpu
        if haskey(MEANPOOL_IMSIZE, name)
            @test_broken r.output <= GPU_RTOL
            @test gpu_cpu_error(name, MEANPOOL_IMSIZE[name]) <= GPU_RTOL
        else
            @test r.output <= GPU_RTOL
            @test r.features <= GPU_RTOL
        end
    end
end

@testset "gradient" begin
    for name in ["mobilenetv4_conv_blur_medium", "vgg_adaptive_w", "gmlp"]
        model = GPU(fixture_model(fixture(name); eltype = Float32))
        @test check_gradient(model, first(gpu_inputs(name))...)
    end
    model = GPU(fixture_model(fixture("res2net"); eltype = Float32))
    @test check_gradient(model, GPU(randn(Float32, 64, 64, 3, 2)))
end

@testset "bfloat16" begin
    for name in ["resnet_basic", "vit_tiny"]
        model = GPU(fixture_model(fixture(name); eltype = Float32))
        @test check_half_precision(model, first(gpu_inputs(name))...; T = BFloat16,
            rtol = BF16_RTOL)
    end
end
