using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup
using Metalhead.Transformers: get_text_features, get_vision_features, Sam3ImageProcessor,
                              post_process_object_detection,
                              post_process_instance_segmentation,
                              post_process_semantic_segmentation

const SAM3_FIXTURES = ["sam3_tiny", "sam3_boxes", "sam3_options"]

f32(x) = x isa AbstractArray{Float64} ? Float32.(x) : x
function fixture_args(name)
    acts = Metalhead.Checkpoint(joinpath(fixture(name), "acts.safetensors"))
    args, kwargs = fixture_inputs(acts)
    return map(f32, args), map(f32, kwargs)
end

@testset "fixtures" begin
    for name in SAM3_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.pass
    end
end

@testset "gradient" begin
    for name in ["sam3_boxes", "sam3_options"]
        model = fixture_model(fixture(name); eltype = Float32)
        args, kwargs = fixture_args(name)
        # transformers uses neither the last level of the feature pyramid nor the
        # projection of CLIP
        unused = (model.vision_encoder.neck.fpn_layers[end],
            model.text_encoder.text_projection)
        @test check_gradient(model, args...; unused, kwargs...)
    end
end

@testset "half precision" begin
    # sam3_boxes has reached 1.3e-2 in CI
    for name in SAM3_FIXTURES
        @test check_half_precision(name; rtol = 2e-2)
    end
end

const TINY_BACKBONE = (; hidden_size = 16, intermediate_size = 24, num_hidden_layers = 2,
    num_attention_heads = 2, image_size = 32, patch_size = 4, window_size = 4,
    global_attn_indexes = (1,), pretrain_image_size = 16)
const TINY_BLOCK = (; hidden_size = 16, num_layers = 1, num_attention_heads = 2,
    intermediate_size = 24)
const TINY = (;
    vision_config = (; backbone_config = TINY_BACKBONE, fpn_hidden_size = 16),
    text_config = (; vocab_size = 40, hidden_size = 16, intermediate_size = 24,
        projection_dim = 8, num_hidden_layers = 1, num_attention_heads = 2,
        max_position_embeddings = 6),
    geometry_encoder_config = TINY_BLOCK, detr_encoder_config = TINY_BLOCK,
    detr_decoder_config = (; TINY_BLOCK..., num_layers = 2, num_queries = 4),
    mask_decoder_config = (; hidden_size = 16, num_upsampling_stages = 2,
        num_attention_heads = 2))

@testset "config" begin
    @test issubset(["Sam3Model", "Sam3VideoModel"], list_models(; byfamily = true)["sam3"])
    model = create_model("Sam3Model"; TINY...)
    pixel_values = randn(Float32, 32, 32, 3, 2)
    input_ids = rand(1:39, 6, 2)
    attention_mask = [ones(Int, 6) [ones(Int, 4); 0; 0]]
    y = model(pixel_values, input_ids; attention_mask)
    @test keys(y) ==
          (:pred_masks, :pred_boxes, :pred_logits, :presence_logits, :semantic_seg)
    @test size(y.pred_masks) == (32, 32, 4, 2)
    @test size(y.pred_boxes) == (4, 4, 2)
    @test size(y.pred_logits) == (4, 2)
    @test size(y.presence_logits) == (1, 2)
    @test size(y.semantic_seg) == (32, 32, 1, 2)

    # the features of the images and of the prompts can be reused
    vision = get_vision_features(model, pixel_values)
    text = get_text_features(model, input_ids; attention_mask)
    @test size(text.pooler_output) == (16, 6, 2)
    @test model(vision, text; attention_mask).pred_masks ≈ y.pred_masks

    # padded box prompts are masked: an image's padding doesn't change its detections
    boxes = Float32[0.5 0.4; 0.5 0.6; 0.3 0.2; 0.4 0.5;;;]
    one = model(pixel_values[:, :, :, 1:1], input_ids[:, 1:1];
        input_boxes = boxes[:, 1:1, :], input_boxes_labels = [1;;])
    padded = model(pixel_values[:, :, :, 1:1], input_ids[:, 1:1]; input_boxes = boxes,
        input_boxes_labels = [1; -10;;])
    @test padded.pred_logits ≈ one.pred_logits
    @test padded.pred_masks ≈ one.pred_masks
    @test model(pixel_values[:, :, :, 1:1], input_ids[:, 1:1];
        input_boxes = boxes[:, 1:1, :]).pred_masks ≈ one.pred_masks  # positive by default
    # a single prompt is repeated for the box prompts of each image
    both = model(vision, get_text_features(model, input_ids[:, 1:1]);
        input_boxes = cat(boxes[:, 1:1, :], boxes[:, 1:1, :]; dims = 3))
    @test both.pred_logits[:, 1] ≈ model(vision, get_text_features(model,
        input_ids[:, [1, 1]]); input_boxes = cat(boxes[:, 1:1, :], boxes[:, 1:1, :];
        dims = 3)).pred_logits[:, 1]

    # the video model runs its detector on images
    video = create_model("Sam3VideoModel"; detector_config = TINY)
    @test video.detector_model isa Metalhead.Transformers.Sam3Model
    vy = video(pixel_values, input_ids; attention_mask)
    @test size(vy.pred_masks) == size(y.pred_masks)
    @test vy.pred_logits ≈
          video.detector_model(pixel_values, input_ids; attention_mask).pred_logits

    build(; backbone = (;), vision = (;), kw...) = create_model("Sam3Model"; TINY...,
        vision_config = (; TINY.vision_config..., vision...,
            backbone_config = (; TINY_BACKBONE..., backbone...)), kw...)
    theta(backbone, i) = build(;
        backbone).vision_encoder.backbone.layers[i].rotary_emb.rope_theta
    @test theta((; rope_theta = 50.0), 1) == 50
    @test theta((; rope_parameters = (; rope_type = "axial", rope_theta = 7)), 2) == 7
    @test_throws "rope_type" build(;
        backbone = (; rope_parameters = (; rope_type = "default")))
    @test_throws "scale_factor" build(; vision = (; scale_factors = (3.0, 1.0)))
    @test_throws "divisible by 4" build(;
        backbone = (; hidden_size = 12, num_attention_heads = 2))
    @test_throws "hidden_act" build(; backbone = (; hidden_act = "tanh"))
    @test_throws "patch_size" build(; backbone = (; patch_size = (4, 4)))
    @test_throws "divisible by 8" build(;
        mask_decoder_config = (; TINY.mask_decoder_config..., hidden_size = 12))
    @test_throws "global attention" model(randn(Float32, 36, 36, 3, 1), input_ids[:, 1:1])
    @test_throws "max_position_embeddings" model(pixel_values, rand(1:39, 7, 2))
    @test_throws "pad on the right" model(pixel_values, input_ids;
        attention_mask = [0 1; 1 1; 1 1; 1 1; 1 1; 1 1])
    @test_throws "input_boxes_labels" model(pixel_values, input_ids; input_boxes = boxes,
        input_boxes_labels = [1 1])
end

@testset "processor inputs and post-processing" begin
    model = create_model("Sam3Model"; TINY...)
    pixel_values = randn(Float32, 32, 32, 3, 2)
    input_ids = rand(1:39, 6, 2)
    attention_mask = [ones(Int, 6) [ones(Int, 4); 0; 0]]
    boxes = Float32[0.5 0.4; 0.5 0.6; 0.3 0.2; 0.4 0.5;;;]
    labels = [1; -10;;]
    # the output of a processor, whose `original_sizes` the model ignores
    inputs = (; pixel_values = pixel_values[:, :, :, 1:1],
        original_sizes = ((32, 32),), input_ids = input_ids[:, 1:1],
        attention_mask = attention_mask[:, 1:1], input_boxes = boxes,
        input_boxes_labels = labels)
    y = model(inputs)
    @test y.pred_masks ≈ model(inputs.pixel_values, inputs.input_ids;
        inputs.attention_mask, input_boxes = boxes, input_boxes_labels = labels).pred_masks
    @test model((; pixel_values, input_ids, attention_mask)).pred_logits ≈
          model(pixel_values, input_ids; attention_mask).pred_logits
    video = create_model("Sam3VideoModel"; detector_config = TINY)
    @test video(inputs).pred_logits ≈ video.detector_model(inputs).pred_logits
    @test_throws "input_ids" model((; pixel_values))
    @test_throws "has no `input_ids` (it has `pixel_values`)" model((; pixel_values))
    @test_throws "pixel_values" model((; input_ids))

    ip = Sam3ImageProcessor()
    logits = [3.0f0 -1; 0 2; -3 0; 1 -2]
    outputs = (; pred_logits = logits, presence_logits = [0.0f0 4],
        pred_boxes = repeat(Float32[0.1, 0.2, 0.5, 0.6], 1, 4, 2),
        pred_masks = repeat([10.0f0 -10; -10 10], 4, 4, 4, 2),
        semantic_seg = fill(1.0f0, 8, 8, 1, 2))
    scores = Metalhead.NNlib.sigmoid.(logits) .* Metalhead.NNlib.sigmoid.([0.0f0 4])
    detections = post_process_object_detection(ip, outputs; threshold = 0.4)
    @test detections[1].scores ≈ scores[scores[:, 1] .> 0.4, 1]
    @test length(detections[2].scores) == 2
    @test size(detections[1].boxes) == (4, length(detections[1].scores))
    scaled = post_process_object_detection(ip, outputs; threshold = 0.4,
        target_sizes = [(10, 20), (30, 40)])
    @test scaled[1].boxes[:, 1] ≈ [2, 2, 10, 6]
    @test_throws "target sizes" post_process_object_detection(ip, outputs;
        target_sizes = [(10, 20)])
    instances = post_process_instance_segmentation(ip, outputs; threshold = 0.4,
        target_sizes = [(16, 12), (8, 8)])
    @test size(instances[1].masks) == (12, 16, length(instances[1].scores))
    @test eltype(instances[1].masks) == Bool
    @test instances[2].masks[:, :, 1] == repeat([true false; false true], 4, 4)
    none = post_process_instance_segmentation(ip, outputs; threshold = 0.99)
    @test size(none[1].masks) == (8, 8, 0)
    semantic = post_process_semantic_segmentation(ip, outputs;
        target_sizes = [(4, 6), (8, 8)])
    @test size(semantic[1].segmentation) == (6, 4)
    @test all(semantic[2].segmentation)
    @test semantic[2].scores ≈ fill(Metalhead.NNlib.sigmoid(1.0f0), 8, 8)
end

if TEST_NETWORK
    @testset "references" begin
        # SAM 3's checkpoint (jetjodh/sam3) is 3.4 GB
        files = [reference_files_of("Sam3Model");
                 TEST_LARGE ? reference_files_of("Sam3VideoModel") : String[]]
        for file in files
            r = check_sam3_reference(file)
            @test r.logits <= 1e-4
            @test r.boxes <= 1e-4
            @test r.presence <= 1e-4
            @test r.masks <= 1e-4
            @test r.semantic <= 1e-4
            @test r.post <= 1e-4
            @test r.post_same
        end
    end
end
