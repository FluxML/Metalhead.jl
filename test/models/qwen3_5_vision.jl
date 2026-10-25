using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

const QWEN3_5_VISION_FIXTURES = ["qwen3_5_vision_tiny", "qwen3_5_vision_options"]

fixture_acts(name) = Metalhead.Checkpoint(joinpath(fixture(name), "acts.safetensors"))

@testset "fixtures" begin
    for name in QWEN3_5_VISION_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.pass
    end
end

@testset "gradient" begin
    for name in QWEN3_5_VISION_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        acts = fixture_acts(name)
        @test check_gradient(model, Float32.(acts["inputs/1"]), acts["inputs/2"])
    end
end

@testset "half precision" begin
    for name in QWEN3_5_VISION_FIXTURES
        @test check_half_precision(name)
    end
end

@testset "config" begin
    tiny = (; depth = 2, hidden_size = 16, intermediate_size = 24, num_heads = 2,
        patch_size = 2, out_hidden_size = 12, num_position_embeddings = 16)
    model = create_model("Qwen3_5VisionModel"; tiny...)
    @test "Qwen3_5VisionModel" in list_models(; byfamily = true)["qwen"]
    @test model.rotary_pos_emb.rope_theta == 10_000
    grid_thw = [1 2 1; 4 2 6; 6 4 2]
    pixel_values = randn(Float32, 3 * 2 * 2 * 2, 24 + 16 + 12)
    y = model(pixel_values, grid_thw)
    @test size(y.last_hidden_state) == (16, 52)
    @test size(y.pooler_output) == (12, 13)

    # each frame of each image attends to its own patches only
    ranges = (1:24, 25:40, 41:52)
    alone = [model(pixel_values[:, r], grid_thw[:, i:i]) for (i, r) in enumerate(ranges)]
    @test reduce(hcat, [a.pooler_output for a in alone]) ≈ y.pooler_output
    @test model(pixel_values[:, 25:40], [1 1; 2 2; 4 4]).pooler_output ≈
          alone[2].pooler_output

    build(; kw...) = create_model("Qwen3_5VisionModel"; tiny..., kw...)
    theta(; kw...) = build(; kw...).rotary_pos_emb.rope_theta
    @test theta(rope_theta = 50) == 50
    @test theta(rope_parameters = Dict("rope_type" => "default", "rope_theta" => 7)) == 7
    @test theta(rope_parameters = (; rope_theta = 3.0)) == 3
    @test theta(deepstack_visual_indexes = ()) == 10_000
    @test_throws "rope_type" build(
        rope_parameters = Dict("rope_type" => "linear", "factor" => 2.0))
    @test_throws "deepstack" build(deepstack_visual_indexes = (1,))
    @test_throws "hidden_act" build(hidden_act = "gelu")
    @test_throws "head_dim" build(head_dim = 4)
    @test_throws "divisible" build(num_heads = 3)
    @test_throws "divisible by 4" build(hidden_size = 12, num_heads = 2)
    @test_throws "patches" model(pixel_values[:, 1:50], grid_thw)
    @test_throws "spatial_merge_size" model(pixel_values[:, 1:12], [1; 3; 4;;])
    @test_throws "size (3, num_images)" model(pixel_values, grid_thw[2:3, :])
end

@testset "processor inputs" begin
    # the output of the image processor, as it is; its own tests are in
    # test/processors/qwen3_vl.jl
    model = create_model("Qwen3_5VisionModel"; depth = 1, hidden_size = 16,
        intermediate_size = 24, num_heads = 2, patch_size = 4, out_hidden_size = 12,
        num_position_embeddings = 16)
    ip = Metalhead.Transformers.Qwen2VLImageProcessor(; patch_size = 4,
        size = (; shortest_edge = 1, longest_edge = 4096))
    inputs = ip([rand(UInt8, 30, 20, 3), rand(UInt8, 16, 16, 3)])
    y = model(inputs)
    @test size(y.pooler_output) == (12, sum(prod, eachcol(inputs.image_grid_thw)) ÷ 4)
    @test y.pooler_output == model(inputs.pixel_values, inputs.image_grid_thw).pooler_output
    @test model(merge(inputs, (; input_ids = [1, 2]))).pooler_output == y.pooler_output
    @test_throws "has no `image_grid_thw`" model((; inputs.pixel_values))
end
