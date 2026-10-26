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

# The outputs of a processor move to the GPU but for `original_sizes`, which the
# post-processing reads on the CPU.
@testset "processor outputs" begin
    sam3 = joinpath(pkgdir(Metalhead), "test", "fixtures", "processors", "unsloth--sam3.1")
    proc = create_processor(sam3; size = (; height = 40, width = 40))
    images = [rand(UInt8, 50, 30, 3), rand(UInt8, 20, 60, 3)]
    inputs = GPU(proc(; images))
    @test inputs.pixel_values isa CUDA.CuArray
    @test inputs.original_sizes == ((30, 50), (60, 20))
    # the tiny model's own prompts, as its vocabulary and positions are smaller
    model = GPU(fixture_model(fixture("sam3_tiny"); eltype = Float32))
    input_ids = GPU([1 2; 5 7; 9 3; 2 2])
    outputs = model(merge(inputs, (; input_ids)))
    objects = Metalhead.Transformers.post_process_instance_segmentation(proc, outputs;
        threshold = 0.0, target_sizes = inputs.original_sizes)
    @test size(objects[2].masks)[1:2] == (20, 60)
end
