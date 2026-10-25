using Metalhead, Test
using JSON: JSON
using SafeTensors: SafeTensors
using Metalhead.Transformers: Sam3Processor, Sam3ImageProcessor, TokenizersBackend,
                              post_process_object_detection,
                              post_process_instance_segmentation,
                              post_process_semantic_segmentation

include(joinpath(@__DIR__, "..", "processors_setup.jl"))

# Written by porting/dump_processor.py, through transformers' AutoProcessor
const PROCESSORS = joinpath(pkgdir(Metalhead), "test", "fixtures", "processors")
const SAM3 = joinpath(PROCESSORS, "unsloth--sam3.1")
const SAM3_VIDEO = joinpath(PROCESSORS, "jetjodh--sam3")
const FILE = joinpath(pkgdir(Metalhead), "test", "references", "processors",
    "sam3.safetensors")
const META = SafeTensors.deserialize(FILE).metadata
const REF = Metalhead.Checkpoint(FILE)
const CASES = sort!([k for (k, v) in META
                     if startswith(v, "{") && haskey(JSON.parse(v), "create")])

numbered(prefix) = [REF["$prefix/$i"] for i in 1:count(startswith("$prefix/"), keys(REF))]

# The processor and the inputs of a case, with its prompts as token ids (`strings = false`)
# or as strings. Python's float images are already rescaled (`do_rescale = False`), which
# Metalhead's images of reals are.
function case(name; strings = false)
    m = JSON.parse(META[name])
    create = Metalhead.jsonvalue(m["create"])
    name == "float" && (create = Base.structdiff(create, (; do_rescale = false)))
    ids = numbered("$name/text_ids")
    text = strings ? m["text"] : isempty(ids) ? nothing : ids
    # strings need the tokenizer.json of the repository
    source = strings ? m["repo"] : m["repo"] == "jetjodh/sam3" ? SAM3_VIDEO : SAM3
    proc = create_processor(source; create...)
    inputs = (; images = numbered("$name/image"), text, input_boxes = m["boxes"],
        input_boxes_labels = m["labels"])
    return proc, inputs
end

# Whether the outputs of the processor are transformers': exact, but for images of reals.
function check_outputs(name, out)
    exact = !(name in ("float", "float_no_rescale"))
    expected = Set(split(k, '/')[2] for k in keys(REF)
                   if startswith(k, "$name/") && !occursin(r"/(image|text_ids)/", k))
    @test Set(string.(keys(out))) == expected
    for k in keys(out)
        r = REF["$name/$k"]
        if k === :original_sizes
            @test out[k] == Tuple((h, w) for (h, w) in eachcol(r))
        elseif k === :input_ids
            @test out[k] == r .+ 1
        elseif k === :pixel_values
            @test eltype(out[k]) == Float32
            @test size(out[k]) == size(r)
            @test exact ? out[k] == r :
                  maximum(abs.(out[k] .- r)) <= 1.0e-6 * maximum(abs, r)
        else
            @test out[k] == r
        end
    end
end

@testset "loading" begin
    proc = @test_logs create_processor(SAM3)
    @test proc isa Sam3Processor
    @test proc.image_processor isa Sam3ImageProcessor
    @test proc.tokenizer isa TokenizersBackend
    @test proc.point_pad_value == -10
    ip = proc.image_processor
    @test ip.size == (; height = 1008, width = 1008)
    @test ip.resample == 2 && ip.rescale_factor == 1 / 255
    @test ip.image_mean == ip.image_std == (0.5, 0.5, 0.5)
    tok = proc.tokenizer
    @test tok.tokenizer === nothing
    @test tok.tokenizer_class == "CLIPTokenizer"
    @test tok.pad_token == "<|endoftext|>" && tok.bos_token == "<|startoftext|>"
    @test tok.added_tokens_decoder == Dict(49406 => "<|startoftext|>", 49407 => "<|endoftext|>")
    @test tok.model_max_length == 32 && tok.padding_side == "right"
    # jetjodh/sam3 names Sam3VideoProcessor, whose detector of images is Sam3Processor
    video = @test_logs create_processor(SAM3_VIDEO)
    @test video isa Sam3Processor
    @test Metalhead.settings(video.image_processor) == Metalhead.settings(ip)
    @test JSON.parse(META["video_processor"])["cls"] == "Sam3VideoProcessor"
end

@testset "transformers' processor" begin
    @test length(CASES) == 18
    for name in CASES
        proc, inputs = case(name)
        @testset "$name" begin
            check_outputs(name, proc(; inputs...))
        end
    end
    # the image processor alone
    proc, inputs = case("batch")
    out = proc.image_processor(inputs.images)
    @test keys(out) == (:pixel_values, :original_sizes)
    @test out.pixel_values == REF["batch/pixel_values"]
    @test proc.image_processor(inputs.images[1]).pixel_values ==
          REF["batch/pixel_values"][:, :, :, 1:1]
    # boxes with the sizes of the images instead of the images
    out = proc(; original_sizes = [(9, 13), (16, 24), (30, 8)], inputs.input_boxes,
        inputs.input_boxes_labels, text = inputs.text)
    @test out.input_boxes == REF["batch/input_boxes"]
    @test out.input_ids == REF["batch/input_ids"] .+ 1
    @test !haskey(out, :pixel_values)
    # boxes as an array (4, num_boxes, num_images) and labels (num_boxes, num_images)
    boxes = Float32[1 0; 2 0; 5 20; 7 10;;; 0 3.5; 0 4; 20 12; 10 15.25]
    out2 = proc(; images = inputs.images[1:2], input_boxes = boxes,
        input_boxes_labels = [1 0; 1 1], text = inputs.text[1:2])
    @test out2.input_boxes[:, :, 2] == REF["batch/input_boxes"][:, :, 2]
    @test out2.input_boxes_labels == [1 0; 1 1]
    # one prompt for the boxes of several images, which the model repeats
    out = proc(; images = inputs.images[1:2], text = inputs.text[1],
        input_boxes = [[(1, 2, 5, 7)], [(0, 0, 20, 10)]])
    @test size(out.input_ids) == (32, 1)
    @test out.input_boxes_labels == [1 1]
    @test_throws "original_sizes" proc(; input_boxes = [[(1, 2, 5, 7)]])
    @test_throws "2 prompts" proc(; images = inputs.images[1:1], text = inputs.text[1:2],
        input_boxes = [[(1, 2, 5, 7)]])
    @test_throws "boxes" proc(; images = inputs.images[1:1], text = inputs.text[1:1],
        input_boxes = [[(1, 2, 5, 7)]], input_boxes_labels = [[1, 0]])
    @test_throws "(x1, y1, x2, y2)" proc(; images = inputs.images[1:1],
        text = inputs.text[1:1], input_boxes = [[(1, 2, 5)]])
    @test_throws "`nothing`" proc(; images = inputs.images[1:1], text = [nothing])
    @test_throws "different sizes" create_processor(SAM3; do_resize = false)(;
        images = inputs.images)
    # prompts of more than 32 tokens are kept, as in transformers, but can't be batched
    long = only(numbered("long/text_ids"))
    @test size(proc(; text = [long]).input_ids) == (length(long), 1)
    @test_throws "truncat" proc(; text = [long, inputs.text[1]])
    @test_throws "channels" create_processor(SAM3; do_convert_rgb = false)(;
        images = numbered("grey/image"))
end

@testset "documented outputs" begin
    outputs = map(name -> (((proc, inputs) = case(name)); proc(; inputs...)), CASES)
    check_documented_keys(Sam3Processor, outputs)
    proc, inputs = case("batch")
    check_documented_keys(Sam3ImageProcessor, [proc.image_processor(inputs.images)])
end

@testset "post-processing" begin
    proc = create_processor(SAM3)
    outputs = (; (Symbol(k) => REF["post/outputs/$k"]
                  for k in ("pred_logits", "pred_boxes", "presence_logits", "pred_masks",
        "semantic_seg"))...)
    settings = JSON.parse(META["post"])
    target_sizes = [Tuple(s) for s in settings["target_sizes"]]
    threshold = settings["threshold"]
    result(name, i, k) = REF["post/$name/$i/$k"]
    for (name, sizes) in ("" => target_sizes, "_normalized" => nothing)
        detections = post_process_object_detection(proc, outputs; threshold,
            target_sizes = sizes)
        instances = post_process_instance_segmentation(proc, outputs; threshold,
            mask_threshold = settings["mask_threshold"], target_sizes = sizes)
        semantic = post_process_semantic_segmentation(proc, outputs;
            threshold = settings["semantic_threshold"], target_sizes = sizes)
        for i in 1:2
            d, s = detections[i], instances[i]
            @test d.scores ≈ result("detection$name", i, "scores")
            @test d.boxes ≈ result("detection$name", i, "boxes")
            @test s.scores ≈ result("instance$name", i, "scores")
            @test s.boxes ≈ result("instance$name", i, "boxes")
            @test s.masks == result("instance$name", i, "masks")
            @test semantic[i].segmentation == result("semantic$name", i, "segmentation")
            @test semantic[i].segmentation_scores ≈
                  result("semantic$name", i, "segmentation_scores")
            # transformers' keys
            @test keys(d) == (:scores, :boxes)
            @test keys(s) == (:scores, :boxes, :masks)
            @test keys(semantic[i]) == (:segmentation, :segmentation_scores)
        end
    end
    # the image processor post-processes as well, and target sizes may be a matrix
    @test post_process_object_detection(proc.image_processor, outputs; threshold,
        target_sizes = [12 9; 16 7])[2].boxes ≈ result("detection", 2, "boxes")
    @test_throws "target sizes" post_process_object_detection(proc, outputs;
        target_sizes = target_sizes[1:1])
    none = post_process_instance_segmentation(proc, outputs; threshold = 0.999,
        target_sizes)
    @test size(none[1].masks) == (16, 12, 0)
end

@testset "settings" begin
    # overrides by name, merged into nested settings
    proc = @test_logs create_processor(SAM3; size = (; height = 16), point_pad_value = -5,
        padding_side = "left")
    @test proc.image_processor.size == (; height = 16, width = 1008)
    @test proc.point_pad_value == -5
    @test proc.tokenizer.padding_side == "left"
    text = sprint(show, MIME("text/plain"), proc)
    @test occursin(r"size = \(height = 16, width = 1008\) +override \(processor_config\.json: \(height = 1008, width = 1008\)\)",
        text)
    @test occursin(r"point_pad_value = -5 +override \(processor_config\.json: -10\)", text)
    @test occursin(r"point_pad_value = -5 +override \(class default: -10\)",
        sprint(show, MIME("text/plain"),
            create_processor(SAM3_VIDEO; point_pad_value = -5)))
    @test occursin(r"resample = 2 \(bilinear\) +processor_config\.json", text)
    @test occursin(r"padding_side = \"left\" +override \(class default: \"right\"\)", text)
    @test occursin(r"sep_token = nothing +class default", text)
    @test occursin("loaded from tokenizer_file when a string needs it", text)
    @test occursin("image_processor: Sam3ImageProcessor", text)
    ip = proc.image_processor
    changed = Sam3ImageProcessor(; Metalhead.settings(ip)..., do_resize = false,
        provenance = ip.provenance)
    @test occursin(r"do_resize = false +set after loading",
        sprint(show, MIME("text/plain"), changed))
    @test startswith(sprint(show, ip), "Sam3ImageProcessor(do_resize = true, size = ")
    @test startswith(sprint(show, proc),
        "Sam3Processor(image_processor = Sam3ImageProcessor(")
    # by hand, over the class defaults
    @test Sam3ImageProcessor().size == (; height = 1008, width = 1008)
    swapped = Sam3Processor(Sam3ImageProcessor(; size = (; height = 16, width = 24)),
        proc.tokenizer)
    _, inputs = case("exact")
    @test swapped(; inputs...).pixel_values == REF["exact/pixel_values"]
    @test occursin("Sam3ImageProcessor\n", sprint(show, MIME("text/plain"), swapped))

    # an override that no component has, or that a component ignores, or unsupported
    @test_throws "no component of Sam3Processor has a setting `reesample` (Sam3ImageProcessor has `resample`)" create_processor(SAM3;
        reesample = 3)
    @test_throws "`mask_size` is not used by Sam3ImageProcessor" create_processor(SAM3;
        mask_size = (; height = 1, width = 1))
    @test_throws "`do_pad = true` is not supported by Sam3ImageProcessor" create_processor(SAM3;
        do_pad = true)
    @test_throws "`resample = 1` (lanczos) is not supported" create_processor(SAM3;
        resample = 1)
    @test_throws "`size` must have `height` and `width` only" create_processor(SAM3;
        size = (; shortest_edge = 3))
    @test_throws "Sam3ImageProcessor has no setting `foo`" Metalhead.processor_from_entries(
        Sam3ImageProcessor, []; foo = 1)

    # the files, strictly: an unknown key warns once, an unsupported value is an error
    function with_config(f, edit)
        mktempdir() do dir
            for file in readdir(SAM3)
                cp(joinpath(SAM3, file), joinpath(dir, file))
            end
            config = JSON.parsefile(joinpath(dir, "processor_config.json"))
            edit(config)
            write(joinpath(dir, "processor_config.json"), JSON.json(config))
            return f(dir)
        end
    end
    with_config(c -> c["image_processor"]["sam3_test_key"] = 1) do dir
        @test_logs (:warn,
            "`sam3_test_key` in processor_config.json is not a setting of Sam3ImageProcessor; it is ignored") create_processor(dir)
        @test_logs create_processor(dir)  # once
    end
    with_config(c -> c["image_processor"]["do_center_crop"] = true) do dir
        @test_throws "`do_center_crop = true` in processor_config.json is not supported" create_processor(dir)
    end
    with_config(c -> c["image_processor"]["do_center_crop"] = nothing) do dir
        @test create_processor(dir) isa Sam3Processor  # null is missing
    end
    with_config(c -> c["image_processor"]["resample"] = 0) do dir
        @test_throws "`resample = 0` (nearest) is not supported (supported: 2 (bilinear), 3 (bicubic))" create_processor(dir)
    end
    sam2 = c -> c["image_processor"]["image_processor_type"] = "Sam2ImageProcessorFast"
    with_config(sam2) do dir
        @test_throws "`Sam2ImageProcessorFast`" create_processor(dir)
    end
    with_config(c -> c["processor_class"] = "Sam2Processor") do dir
        @test_throws "the processor class `Sam2Processor` (`processor_class` in processor_config.json) has not been ported" create_processor(dir)
    end
end

# Strings through HuggingFaceTokenizers.jl, which METALHEAD_TEST_NETWORK adds to the test
# environment (runtests.jl), with the tokenizer.json of the repositories
const TEST_NETWORK = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))
if TEST_NETWORK
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers
    @testset "strings" begin
        for name in CASES
            proc, inputs = case(name; strings = true)
            @testset "$name" begin
                check_outputs(name, proc(; inputs...))
            end
        end
        proc = create_processor("unsloth/sam3.1")
        @test Metalhead.Transformers.decode(proc, REF["boxes_only/input_ids"][1:3] .+ 1) ==
              "<|startoftext|>visual <|endoftext|>"
    end
end
