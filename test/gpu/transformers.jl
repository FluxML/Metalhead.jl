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
    @test inputs.original_sizes == ((; height = 30, width = 50), (; height = 60, width = 20))
    # the tiny model's own prompts, as its vocabulary and positions are smaller
    model = GPU(fixture_model(fixture("sam3_tiny"); eltype = Float32))
    input_ids = GPU([1 2; 5 7; 9 3; 2 2])
    outputs = model(merge(inputs, (; input_ids)))
    objects = post_process_instance_segmentation(proc, outputs;
        threshold = 0.0, target_sizes = inputs.original_sizes)
    @test size(objects[2].masks)[1:2] == (20, 60)
end

# The vision-language model of Qwen3.5 on the GPU takes a processor's output, a batch
# padded on the right.
@testset "Qwen3.5 processor outputs" begin
    T = Metalhead.Transformers
    model = fixture_model(fixture("qwen3_5_tiny"); eltype = Float32)
    ip = T.Qwen2VLImageProcessor(; patch_size = 2, image_mean = 0.5, image_std = 0.5,
        size = (; shortest_edge = 16, longest_edge = 256))
    proc = T.Qwen3VLProcessor(ip,
        IdsTokenizer(Dict("<|image_pad|>" => 60, "<|video_pad|>" => 61)))
    inputs = proc(; images = [rand(UInt8, 12, 8, 3), rand(UInt8, 9, 7, 3)],
        text = ["5 62 <|image_pad|> 63 7 8", "62 <|image_pad|> 63 9 10 11 12 13 14"])
    @test inputs.attention_mask[end, :] == [0, 1]
    y = model(inputs)
    y_gpu = GPU(model)(GPU(inputs))
    @test y_gpu isa CUDA.CuArray
    @test Array(y_gpu) ≈ y rtol = GPU_RTOL
end
