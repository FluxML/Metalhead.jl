using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup
using Flux: Flux, sigmoid
using Metalhead.Transformers: SiglipModel, Siglip2Model, SiglipTextModel, SiglipVisionModel,
                              Siglip2VisionModel, get_text_features, get_image_features,
                              SiglipImageProcessor, Siglip2ImageProcessor

const SIGLIP_FIXTURES = ["siglip_tiny", "siglip_options", "siglip2_tiny", "siglip2_options"]

fixture_acts(name) = Metalhead.Checkpoint(joinpath(fixture(name), "acts.safetensors"))
function fixture_args(name)
    args, kwargs = fixture_inputs(fixture_acts(name))
    f32(x) = x isa AbstractArray{Float64} ? Float32.(x) : x
    return map(f32, args), kwargs
end

@testset "fixtures" begin
    for name in SIGLIP_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.pass
        # the parameters of transformers' model, all loaded by name
        meta = SafeTensors.deserialize(joinpath(fixture(name), "acts.safetensors")).metadata
        model = fixture_model(fixture(name))
        @test sum(length, Flux.trainables(model)) == parse(Int, meta["num_params"])
    end
end

@testset "gradient" begin
    for name in SIGLIP_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        args, kwargs = fixture_args(name)
        @test check_gradient(model, args...; kwargs...)
    end
end

@testset "half precision" begin
    for name in SIGLIP_FIXTURES
        @test check_half_precision(name)
    end
end

const TEXT = (; vocab_size = 40, hidden_size = 16, intermediate_size = 24,
    num_hidden_layers = 1, num_attention_heads = 2, max_position_embeddings = 6)
const VISION = (; hidden_size = 16, intermediate_size = 24, num_hidden_layers = 1,
    num_attention_heads = 2, image_size = 16, patch_size = 4)
const VISION2 = (; Base.structdiff(VISION, (; image_size = 0))..., num_patches = 16)

@testset "SiglipModel" begin
    @test issetequal(["SiglipModel", "Siglip2Model"], list_models(; byfamily = true)["siglip"])
    model = create_model("SiglipModel"; text_config = TEXT, vision_config = VISION)
    @test model.text_model isa SiglipTextModel && model.vision_model isa SiglipVisionModel
    input_ids = rand(1:40, 6, 3)
    pixel_values = randn(Float32, 16, 16, 3, 2)
    y = model(input_ids, pixel_values)
    @test keys(y) == (:logits_per_image, :logits_per_text, :text_embeds, :image_embeds,
        :text_model_output, :vision_model_output)
    @test size(y.logits_per_image) == (3, 2) && y.logits_per_text == y.logits_per_image'
    @test size(y.text_embeds) == (16, 3) && size(y.image_embeds) == (16, 2)
    @test vec(sum(abs2, y.image_embeds; dims = 1)) ≈ ones(2)
    @test size(y.vision_model_output.last_hidden_state) == (16, 16, 2)
    @test size(y.text_model_output.last_hidden_state) == (16, 6, 3)
    # the logits of each pair, alone
    @test y.logits_per_image[2, 1] ≈ model(input_ids[:, 2:2], pixel_values[:, :, :, 1:1]).logits_per_image[1]

    # the features of texts and images alone, as transformers' `get_*_features`
    text = get_text_features(model, input_ids)
    image = get_image_features(model, pixel_values)
    @test text.pooler_output ≈ y.text_model_output.pooler_output
    @test image.pooler_output ≈ y.vision_model_output.pooler_output
    @test get_text_features(model, (; input_ids)).pooler_output ≈ text.pooler_output
    @test get_image_features(model, (; pixel_values)).pooler_output ≈ image.pooler_output
    @test size(get_text_features(model, input_ids[:, 1]).pooler_output) == (16, 1)
    @test model((; input_ids, pixel_values)).logits_per_image ≈ y.logits_per_image
    @test_throws "has no `input_ids` (it has `pixel_values`)" model((; pixel_values))
    @test_throws "get_image_features" model((; pixel_values))
    @test_throws "pixel_values" get_image_features(model, (; input_ids))

    # padding that the mask excludes changes nothing of the other tokens, but the pooled last
    # token is the padding, as in transformers
    mask = [ones(Int, 4, 3); zeros(Int, 2, 3)]
    masked = get_text_features(model, input_ids; attention_mask = mask)
    @test masked.last_hidden_state[:, 1:4, :] ≈
          get_text_features(model, input_ids[1:4, :]).last_hidden_state
    @test_throws "max_position_embeddings" model(rand(1:40, 7, 1), pixel_values)
    @test_throws "pad on the right" get_text_features(model, input_ids;
        attention_mask = [zeros(Int, 1, 3); ones(Int, 5, 3)])

    # images of another size need `interpolate_pos_encoding`, unless square with as many
    # patches
    @test_throws "interpolate_pos_encoding" model(input_ids, randn(Float32, 24, 8, 3, 1))
    other = model(input_ids, randn(Float32, 24, 8, 3, 1); interpolate_pos_encoding = true)
    @test size(other.vision_model_output.last_hidden_state) == (16, 12, 1)
    @test model(input_ids, pixel_values; interpolate_pos_encoding = true).logits_per_image ≈
          y.logits_per_image

    # configurations
    build(; text = (;), vision = (;)) = create_model("SiglipModel";
        text_config = (; TEXT..., text...), vision_config = (; VISION..., vision...))
    @test size(build(; text = (; hidden_size = 8, projection_size = 16)).text_model.head.weight) ==
          (16, 8)
    @test_throws "projection_size" build(; text = (; projection_size = 8))
    @test_throws "vision_use_head" build(; vision = (; vision_use_head = false))
    @test_throws "hidden_act" build(; text = (; hidden_act = "tanh"))
    @test_throws "divisible" build(; vision = (; num_attention_heads = 3))
    @test_throws "integers" build(; vision = (; patch_size = (4, 4)))
    # the config.json of a checkpoint names the model by its `model_type` only
    # (google/siglip2-base-patch16-224), or by `architectures`
    @test Metalhead.find_format(Dict("model_type" => "siglip")) ==
          (Metalhead.Transformers.TRANSFORMERS, "SiglipModel")
    @test Metalhead.find_format(Dict("model_type" => "siglip2", "architectures" => [])) ==
          (Metalhead.Transformers.TRANSFORMERS, "Siglip2Model")
    @test Metalhead.find_format(Dict("model_type" => "siglip",
        "architectures" => ["Siglip2Model"]))[2] == "Siglip2Model"
    mktempdir() do dir
        src = fixture("siglip_tiny")
        cp(joinpath(src, "model.safetensors"), joinpath(dir, "model.safetensors"))
        config = JSON.parsefile(joinpath(src, "config.json"))
        delete!(config, "architectures")
        write(joinpath(dir, "config.json"), JSON.json(config))
        @test create_model(dir) isa SiglipModel
    end
end

@testset "Siglip2Model" begin
    model = create_model("Siglip2Model"; text_config = TEXT, vision_config = VISION2)
    @test model.vision_model isa Siglip2VisionModel
    input_ids = rand(1:40, 6, 2)
    spatial_shapes = [3 4 1; 5 4 3]
    pixel_attention_mask = Int.((1:16) .<= [15 16 3])
    pixel_values = randn(Float32, 48, 16, 3) .* reshape(pixel_attention_mask, 1, 16, 3)
    y = model(input_ids, pixel_values; spatial_shapes, pixel_attention_mask)
    @test size(y.logits_per_image) == (2, 3)
    @test size(y.vision_model_output.last_hidden_state) == (16, 16, 3)
    # the padding of an image changes nothing of its patches and embedding
    alone = model.vision_model(pixel_values[:, 1:3, 3:3]; spatial_shapes = [1; 3;;],
        pixel_attention_mask = ones(Int, 3, 1))
    @test alone.pooler_output ≈ y.vision_model_output.pooler_output[:, 3]
    @test alone.last_hidden_state ≈ y.vision_model_output.last_hidden_state[:, 1:3, 3:3]
    # the inputs of the image processor, as they are
    inputs = (; input_ids, pixel_values, spatial_shapes, pixel_attention_mask)
    @test model(inputs).logits_per_image ≈ y.logits_per_image
    @test get_image_features(model, inputs).pooler_output ≈
          y.vision_model_output.pooler_output
    @test model.vision_model(inputs).pooler_output ≈ y.vision_model_output.pooler_output
    @test get_text_features(model, inputs).pooler_output ≈ y.text_model_output.pooler_output
    @test_throws "has no `spatial_shapes`" model((; input_ids, pixel_values))
    ip = Siglip2ImageProcessor(; patch_size = 4, max_num_patches = 16)
    images = ip([rand(UInt8, 20, 9, 3), rand(UInt8, 7, 30, 3)])
    @test size(model(merge(images, (; input_ids))).logits_per_image) == (2, 2)

    @test_throws "doesn't fit in 16 patches" model(input_ids, pixel_values;
        spatial_shapes = [5 4 1; 4 4 3])
    @test_throws "(2, batch)" model(input_ids, pixel_values;
        spatial_shapes = permutedims(spatial_shapes))
    @test_throws "ones, then zeros" model(input_ids, pixel_values; spatial_shapes,
        pixel_attention_mask = reverse(pixel_attention_mask; dims = 1))
    @test_throws "pixel_attention_mask` has size" model(input_ids, pixel_values;
        spatial_shapes, pixel_attention_mask = pixel_attention_mask[1:15, :])
    @test_throws "square" create_model("Siglip2Model"; text_config = TEXT,
        vision_config = (; VISION2..., num_patches = 15))
    @test_throws "vision_use_head" create_model("Siglip2Model"; text_config = TEXT,
        vision_config = (; VISION2..., vision_use_head = false))
end

if TEST_NETWORK
    # the tokenizer of the processor, which METALHEAD_TEST_NETWORK adds to the test
    # environment (runtests.jl)
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers
    @testset "references" begin
        for file in [reference_files_of("SiglipModel"); reference_files_of("Siglip2Model")]
            r = check_siglip_reference(file)
            @test r.inputs
            @test r.logits <= 1e-4
            @test r.embeds <= 1e-4
            @test r.pooled <= 1e-4
            @test r.hidden <= 1e-4
        end
    end
end
