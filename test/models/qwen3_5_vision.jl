using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup
using Metalhead.Transformers: qwen3_5_image_inputs, smart_resize

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

@testset "image processing" begin
    file = joinpath(pkgdir(Metalhead), "test", "references",
        "qwen2_vl_image_processor.safetensors")
    meta = SafeTensors.deserialize(file).metadata
    ref = Metalhead.Checkpoint(file)

    # the sizes that transformers' `smart_resize` gives with the settings of Qwen3.5
    settings = JSON.parse(meta["smart_resize"])
    for (height, width, h, w) in eachcol(ref["smart_resize"])
        @test smart_resize(height, width; factor = settings["factor"],
            min_pixels = settings["min_pixels"],
            max_pixels = settings["max_pixels"]) == (h, w)
    end
    @test_throws "aspect ratio" smart_resize(10, 2001)

    # transformers' processor on tiny images: exact without resizing (8-bit) or for images
    # of real numbers, else within one level of 255 in a few values, as torchvision resizes
    # 8-bit images in fixed-point arithmetic
    for name in ("exact", "float", "upscale", "downscale", "batch")
        kwargs = JSON.parse(meta[name])
        images = [ref["$name/image/$i"]
                  for i in 1:count(startswith("$name/image/"), keys(ref))]
        std = get(kwargs, "image_std", [0.5])
        pixel_values, grid_thw = qwen3_5_image_inputs(images;
            patch_size = kwargs["patch_size"], merge_size = kwargs["merge_size"],
            temporal_patch_size = kwargs["temporal_patch_size"],
            min_pixels = kwargs["size"]["shortest_edge"],
            max_pixels = kwargs["size"]["longest_edge"],
            image_mean = Tuple(get(kwargs, "image_mean", [0.5, 0.5, 0.5])),
            image_std = Tuple(get(kwargs, "image_std", [0.5, 0.5, 0.5])))
        expected = ref["$name/pixel_values"]
        @test grid_thw == ref["$name/grid_thw"]
        @test eltype(pixel_values) == Float32
        @test size(pixel_values) == size(expected)
        err = abs.(pixel_values .- expected)
        if name == "exact"
            @test pixel_values == expected
        elseif name == "float"
            @test maximum(err) <= 1e-6
        else
            @test maximum(err) <= 1.0001 / (255 * minimum(std))
            @test count(!iszero, err) <= 0.02 * length(err)
        end
    end

    # the patches, merge window by merge window, each `(patch, patch, frame, channel)`
    patch_size, merge_size, temporal_patch_size = 2, 2, 3
    image = [w + 10h + 100c for w in 1:8, h in 1:12, c in 1:3] ./ 1000
    pixel_values, grid_thw = qwen3_5_image_inputs(image; patch_size, merge_size,
        temporal_patch_size, min_pixels = 1, image_mean = (0, 0, 0), image_std = (1, 1, 1))
    @test grid_thw == reshape([1, 6, 4], 3, 1)
    patches = reshape(pixel_values, 2, 2, 3, 3, :)
    windows = [(br, bc, ir, ic) for br in 0:2 for bc in 0:1 for ir in 0:1 for ic in 0:1]
    @test all(enumerate(windows)) do (n, (br, bc, ir, ic))
        row, col = 2br + ir, 2bc + ic
        all(patches[pw, ph, t, c, n] ≈ image[2col + pw, 2row + ph, c]
        for pw in 1:2, ph in 1:2, t in 1:3, c in 1:3)
    end

    # 8-bit images and images of real numbers; the defaults of Qwen3.5
    image = rand(UInt8, 64, 96, 3)
    @test qwen3_5_image_inputs(image; patch_size = 4, min_pixels = 1)[1] ≈
          qwen3_5_image_inputs(image ./ 255; patch_size = 4, min_pixels = 1)[1]
    pixel_values, grid_thw = qwen3_5_image_inputs([rand(UInt8, 640, 480, 3),
        rand(Float32, 100, 300, 3)])
    @test grid_thw == [1 1; 30 28; 40 10]
    @test size(pixel_values) == (16 * 16 * 2 * 3, 1200 + 280)
    @test_throws "channels" qwen3_5_image_inputs(rand(UInt8, 32, 32, 4))

    # the inputs of the model
    model = create_model("Qwen3_5VisionModel"; depth = 1, hidden_size = 16,
        intermediate_size = 24, num_heads = 2, patch_size = 4, out_hidden_size = 12,
        num_position_embeddings = 16)
    pixel_values, grid_thw = qwen3_5_image_inputs(
        [rand(UInt8, 30, 20, 3),
            rand(UInt8, 16, 16, 3)]; patch_size = 4, min_pixels = 1)
    y = model(pixel_values, grid_thw)
    @test size(y.pooler_output) == (12, sum(prod, eachcol(grid_thw)) ÷ 4)
end
