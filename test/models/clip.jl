using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup
using Metalhead.Transformers: CLIPModel, CLIPTextModel, CLIPVisionModel,
                              CLIPTextModelWithProjection, CLIPVisionModelWithProjection,
                              get_text_features, get_image_features

const CLIP_FIXTURES = ["clip_tiny", "clip_options", "clip_text", "clip_text_projection",
    "clip_vision", "clip_vision_projection"]

f32(x) = x isa AbstractArray{Float64} ? Float32.(x) : x
function fixture_args(name)
    acts = Metalhead.Checkpoint(joinpath(fixture(name), "acts.safetensors"))
    args, kwargs = fixture_inputs(acts)
    return map(f32, args), map(f32, kwargs)
end

@testset "fixtures" begin
    for name in CLIP_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.pass
    end
end

@testset "gradient" begin
    for name in CLIP_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        args, kwargs = fixture_args(name)
        @test check_gradient(model, args...; kwargs...)
    end
end

@testset "half precision" begin
    for name in CLIP_FIXTURES
        @test check_half_precision(name)
    end
end

const TINY_TEXT = (; vocab_size = 40, hidden_size = 16, intermediate_size = 24,
    num_hidden_layers = 1, num_attention_heads = 2, max_position_embeddings = 6,
    eos_token_id = 39)
const TINY_VISION = (; hidden_size = 8, intermediate_size = 16, num_hidden_layers = 1,
    num_attention_heads = 2, image_size = 16, patch_size = 8)
const TINY = (; text_config = TINY_TEXT, vision_config = TINY_VISION, projection_dim = 4)

@testset "config" begin
    @test issubset(
        ["CLIPModel", "CLIPTextModel", "CLIPTextModelWithProjection",
            "CLIPVisionModel", "CLIPVisionModelWithProjection"],
        list_models(; byfamily = true)["clip"])
    # transformers' defaults, a ViT-B/32: the parameters of `CLIPModel(CLIPConfig())`, and
    # of the text and vision models alone, with or without their projections
    model = create_model("CLIPModel")
    nparams(m) = sum(length, Metalhead.Flux.trainables(m))
    @test nparams(model) == 151_277_313
    @test length(first(Metalhead.torch_params(model))) == 398
    @test nparams(model.text_model) == 63_165_952
    @test nparams(model.vision_model) == 87_456_000
    @test nparams(model.text_model) + nparams(model.text_projection) == 63_428_096
    @test nparams(model.vision_model) + nparams(model.visual_projection) == 87_849_216
    @test model.logit_scale[] ≈ 2.6592
    @test model.vision_model.embeddings.image_size == 224
    @test model.text_model.encoder.layers[1].mlp.activation_fn ===
          Metalhead.NNlib.quick_gelu
    # older config.json files have `text_config_dict`, which replaces `text_config`: its
    # fields over the defaults of CLIPTextConfig, as transformers reads them
    old = create_model("CLIPModel";
        text_config = (; TINY_TEXT..., eos_token_id = 2, hidden_act = "quick_gelu"),
        text_config_dict = (; TINY_TEXT..., hidden_act = "gelu"),
        vision_config = TINY_VISION, vision_config_dict = nothing, projection_dim = 4)
    @test old.text_model.eos_token_id == 39
    @test old.text_model.encoder.layers[1].mlp.activation_fn === Metalhead.NNlib.gelu_erf
    @test create_model("CLIPModel"; TINY..., vision_config_dict = (; TINY_VISION...,
        patch_size = 4)).vision_model.embeddings.patch_size == 4
    @test create_model("CLIPModel"; text_config = (; TINY_TEXT..., eos_token_id = 2),
        text_config_dict = (; TINY_TEXT...,)).text_model.eos_token_id == 39

    build(; text = (;), vision = (;)) = create_model("CLIPModel"; TINY...,
        text_config = (; TINY_TEXT..., text...), vision_config = (;
            TINY_VISION..., vision...))
    @test_throws "hidden_act" build(; text = (; hidden_act = "tanh"))
    @test_throws "image_size" build(; vision = (; image_size = (16, 16)))
    @test_throws "patch_size" build(; vision = (; patch_size = (8, 8)))
    @test_throws "eos_token_id" build(; text = (; eos_token_id = (39, 2)))
    @test_throws "divisible by `num_attention_heads = 3`" build(;
        vision = (; num_attention_heads = 3))
end

@testset "outputs" begin
    model = create_model("CLIPModel"; TINY...)
    pixel_values = randn(Float32, 16, 16, 3, 2)
    input_ids = [rand(1:38, 5, 3); fill(40, 1, 3)]
    attention_mask = [ones(Int, 6) [ones(Int, 4); 0; 0] ones(Int, 6)]
    y = model(input_ids, pixel_values; attention_mask)
    @test keys(y) == (:logits_per_image, :logits_per_text, :text_embeds, :image_embeds)
    @test size(y.logits_per_image) == (3, 2)
    @test y.logits_per_text == permutedims(y.logits_per_image)
    @test size(y.text_embeds) == (4, 3) && size(y.image_embeds) == (4, 2)
    @test vec(sum(abs2, y.text_embeds; dims = 1)) ≈ ones(3)
    @test vec(sum(abs2, y.image_embeds; dims = 1)) ≈ ones(2)
    # the scaled cosine similarities of the features of the encoders
    text = get_text_features(model, input_ids; attention_mask)
    image = get_image_features(model, pixel_values)
    @test keys(text) == keys(image) == (:last_hidden_state, :pooler_output)
    @test size(text.last_hidden_state) == (16, 6, 3)
    @test size(image.last_hidden_state) == (8, 5, 2)
    cosine(a, b) = sum(a .* b) / sqrt(sum(abs2, a) * sum(abs2, b))
    @test y.logits_per_image[2, 1] ≈
          exp(model.logit_scale[]) *
          cosine(text.pooler_output[:, 2], image.pooler_output[:, 1])
    # the output of a processor, whose other keys are ignored
    inputs = (; input_ids, attention_mask, pixel_values, original_sizes = nothing)
    @test model(inputs).logits_per_image ≈ y.logits_per_image
    @test get_text_features(model, inputs) == text
    @test get_image_features(model, inputs) == image
    @test_throws "pass `text` to the processor" get_text_features(model, (; pixel_values))
    @test_throws "pass `images` to the processor" get_image_features(model, (; input_ids))
    @test_throws "has no `pixel_values` (it has `input_ids`)" model((; input_ids))
    @test_throws "pass `text` and `images` to the processor" model((; pixel_values))
    # the padding, with the end-of-text token, is masked
    seq = [input_ids[1:3, 2:2]; fill(40, 1, 1)]
    padded = [seq; fill(40, 2, 1)]
    mask = attention_mask[:, 2:2]
    @test model(padded, pixel_values; attention_mask = mask).text_embeds ≈
          model(seq, pixel_values).text_embeds
    # images of another size need `interpolate_pos_encoding`, which keeps the embeddings
    # of the checkpoint for its size
    @test_throws "interpolate_pos_encoding = true" model(input_ids, randn(
        Float32, 24, 16, 3, 1))
    @test model(input_ids, pixel_values; interpolate_pos_encoding = true).image_embeds ≈
          y.image_embeds
    @test size(get_image_features(model, randn(Float32, 24, 16, 3, 1);
        interpolate_pos_encoding = true).last_hidden_state) == (8, 7, 1)
    @test_throws "max_position_embeddings" model(rand(1:39, 7, 1), pixel_values)

    # the encoders alone, as their classes are, and on the output of a processor
    for (T, build, key, x, part) in (
        ("CLIPTextModel", TINY_TEXT, :input_ids, input_ids, model.text_model),
        ("CLIPVisionModel", TINY_VISION, :pixel_values, pixel_values, model.vision_model))
        m = Metalhead.Flux.loadmodel!(create_model(T; build..., projection_dim = 4), part)
        out = m(x)
        @test out == m((; key => x))
        @test out.pooler_output ≈ part(x).pooler_output
    end
    text_model = create_model("CLIPTextModelWithProjection"; TINY_TEXT...,
        projection_dim = 4)
    out = text_model((; input_ids, attention_mask))
    @test keys(out) == (:text_embeds, :last_hidden_state) && size(out.text_embeds) == (4, 3)
    @test_throws "pass `text` to the processor" text_model((; pixel_values))
    vision_model = create_model("CLIPVisionModelWithProjection"; TINY_VISION...,
        projection_dim = 4)
    out = vision_model((; pixel_values))
    @test keys(out) == (:image_embeds, :last_hidden_state) &&
          size(out.image_embeds) == (4, 2)
    @test_throws "pass `images` to the processor" vision_model((; input_ids))
end

# Checkpoints of CLIPTextModel and CLIPVisionModel saved before transformers 5 hold their
# weights under `text_model.` and `vision_model.`, which transformers 5 doesn't write
@testset "checkpoints of transformers 4" begin
    for (name, prefix) in ("clip_text" => "text_model", "clip_vision" => "vision_model")
        mktempdir() do dir
            cp(joinpath(fixture(name), "config.json"), joinpath(dir, "config.json"))
            st = SafeTensors.deserialize(joinpath(fixture(name), "model.safetensors"))
            SafeTensors.serialize(joinpath(dir, "model.safetensors"),
                Dict("$prefix.$k" => collect(v) for (k, v) in st))
            args, kwargs = fixture_args(name)
            old = create_model(dir; eltype = Float64)
            new = create_model(fixture(name); eltype = Float64)
            @test old(args...; kwargs...) == new(args...; kwargs...)
        end
    end
end

if TEST_NETWORK
    # the tokenizer of the processor, which METALHEAD_TEST_NETWORK adds to the test
    # environment (runtests.jl)
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers
    @testset "references" begin
        for file in reference_files_of("CLIPModel")
            r = check_clip_reference(file)
            @test r.outputs <= 1e-4
            @test r.features <= 1e-4
            @test r.processor_same
        end
        for file in reference_files_of("CLIPTextModel")
            r = check_clip_reference(file)
            @test r.pooler_output <= 1e-4
            @test r.last_hidden_state <= 1e-4
        end
    end
end
