using Metalhead, Test
using JSON: JSON
using SafeTensors: SafeTensors
using Metalhead.Transformers: CLIPProcessor, CLIPImageProcessor
const T = Metalhead.Transformers

include(joinpath(@__DIR__, "..", "processors_setup.jl"))
include(joinpath(@__DIR__, "..", "data.jl"))
using .TestData: test_data

# Written by porting/dump_processor.py, through transformers' AutoProcessor
const CLIP_REPO = "openai/clip-vit-base-patch32"
const CLIP = joinpath(pkgdir(Metalhead), "test", "fixtures", "processors",
    "openai--clip-vit-base-patch32")
const FILE = test_data("references/processors/clip.safetensors")
const META = SafeTensors.deserialize(FILE).metadata
const REF = Metalhead.Checkpoint(FILE)
const CASES = sort!([k
                     for (k, v) in META
                     if startswith(v, "{") && haskey(JSON.parse(v), "create")])

numbered(prefix) = [REF["$prefix/$i"] for i in 1:count(startswith("$prefix/"), keys(REF))]
settings(name) = JSON.parse(META[name])
function texts(name)
    return (t = settings(name)["text"]; t === nothing ? String[] : t isa String ? [t] : t)
end

# A stand-in for the tokenizer of CLIP, offline: the ids that transformers' tokenizer gives
# the texts of the cases, and its padding token, `<|endoftext|>` in tokenizer_config.json
struct ReferenceTokenizer
    ids::Dict{String, Vector{Int}}
    pad::Union{Nothing, Int}
end
function Metalhead.encode_ids(tok::ReferenceTokenizer, text::AbstractString;
        add_special_tokens::Bool = true)
    ids = tok.ids[text]
    return add_special_tokens ? ids : ids[2:(end - 1)]
end
Metalhead.pad_token_id(tok::ReferenceTokenizer) = tok.pad

const TOKENIZER = ReferenceTokenizer(
    Dict(text => numbered("$name/text_ids")[i] for name in CASES
    for (i, text) in enumerate(texts(name))), 49407)

function with_tokenizer(proc, tok = TOKENIZER)
    return CLIPProcessor(proc.image_processor, tok;
        proc.provenance)
end

# The outputs of the processor of a case: offline, the processor of the fixture files with
# the stand-in for the tokenizer; with `network`, that of the repository, whose tokenizer is
# HuggingFaceTokenizers' of its tokenizer.json. Python's float images are already rescaled
# (`do_rescale = False`), which Metalhead's images of reals are.
function run_case(name; network = false)
    m = settings(name)
    create = Metalhead.jsonvalue(m["create"])
    name == "float" && (create = Base.structdiff(create, (; do_rescale = false)))
    images = numbered("$name/image")
    proc = network ? create_processor(CLIP_REPO; create...) :
           with_tokenizer(create_processor(CLIP; create...))
    return proc(; text = m["text"], images = isempty(images) ? nothing : images)
end

# Whether the outputs of the processor are transformers': exact, but for images of reals.
function check_outputs(name, out)
    m = settings(name)
    exact = !(name in ("float", "float_no_rescale"))
    expected = Set(split(k, '/')[2]
    for k in keys(REF)
    if startswith(k, "$name/") && !occursin(r"/(image|text_ids)/", k))
    @test Set(string.(keys(out))) == expected
    for k in keys(out)
        r = REF["$name/$k"]
        x = out[k]
        if k === :pixel_values
            s = m["stride"]
            s === nothing || (x = x[1:s:end, 1:s:end, :, :])
            @test eltype(x) == Float32
            @test size(x) == size(r)
            @test exact ? x == r : maximum(abs.(x .- r)) <= 1.0e-6 * maximum(abs, r)
        elseif k === :input_ids
            @test x == r .+ 1
        else
            @test x == r
        end
    end
end

@testset "loading" begin
    proc = @test_logs create_processor(CLIP)
    @test proc isa CLIPProcessor
    @test proc.tokenizer === nothing  # the fixture has no tokenizer.json
    ip = proc.image_processor
    @test ip isa CLIPImageProcessor
    # preprocessor_config.json, whose integer sizes are converted as transformers does
    @test ip.size == (; shortest_edge = 224)
    @test ip.crop_size == (; height = 224, width = 224)
    @test ip.resample == 3 && ip.do_center_crop && ip.do_resize && !ip.default_to_square
    @test ip.image_mean == (0.48145466, 0.4578275, 0.40821073)
    @test Metalhead.settings(CLIPImageProcessor()) == Metalhead.settings(ip)
    @test all(settings(name)["cls"] == "CLIPProcessor" for name in CASES)
    @test all(settings(name)["image_cls"] == "CLIPImageProcessor" for name in CASES)
    text = sprint(show, MIME("text/plain"), proc)
    @test occursin(r"size = \(shortest_edge = 224,\) +preprocessor_config\.json", text)
    @test occursin(r"do_convert_rgb = true +class default", text)
    @test endswith(text, "\n  tokenizer: nothing")
end

@testset "transformers' processor" begin
    @test length(CASES) == 20
    for name in CASES
        @testset "$name" begin
            check_outputs(name, run_case(name))
        end
    end
end

@testset "documented outputs" begin
    outputs = [run_case(name) for name in CASES]
    check_documented_keys(CLIPProcessor, outputs)
    images = [o for o in outputs if !haskey(o, :input_ids)]
    @test length(images) == 1
    check_documented_keys(CLIPImageProcessor, images)
end

@testset "texts and images" begin
    proc = with_tokenizer(create_processor(CLIP; size = 24, crop_size = 20))
    images = numbered("batch/image")
    batch = proc(; text = texts("batch"), images)
    # the image processor alone, and one text as a batch of one
    @test proc.image_processor(images) == (; batch.pixel_values)
    @test proc.image_processor(images[1]).pixel_values == batch.pixel_values[:, :, :, 1:1]
    one = proc(; text = texts("batch")[2])
    @test one.input_ids == reshape(TOKENIZER.ids[texts("batch")[2]] .+ 1, :, 1)
    @test keys(proc(; images)) == (:pixel_values,)
    @test_throws "`text` or `images`" proc()
    @test_throws "`text` is a string or a vector of strings" proc(; text = [1, 2])
    # a processor without a tokenizer takes images only, and one without an image
    # processor text only
    ip = proc.image_processor
    @test CLIPProcessor(ip, nothing)(; images) == ip(images)
    @test_throws ["CLIPProcessor has no tokenizer to encode text",
        "`using HuggingFaceTokenizers` before `create_processor`"] CLIPProcessor(ip,
        nothing)(;
        text = "a cat")
    @test_throws "CLIPProcessor has no image processor for `images`" CLIPProcessor(nothing,
        TOKENIZER)(;
        images)
    nopad = CLIPProcessor(nothing, ReferenceTokenizer(TOKENIZER.ids, nothing))
    @test_throws "the tokenizer has no padding token" nopad(; text = texts("batch"))
    unresized = create_processor(CLIP; do_resize = false, do_center_crop = false)
    @test_throws "different sizes" unresized(; images)
    @test_throws "channels" create_processor(CLIP; do_convert_rgb = false)(;
        images = numbered("grey/image"))
    @test_throws "no images" ip([])
    # 8-bit images and images of reals, which resize alike but for the rounding to 8 bits
    image = rand(UInt8, 50, 30, 3)
    @test ip(image).pixel_values ≈ ip(image ./ 255).pixel_values rtol = 0.02
    exact = CLIPImageProcessor(; do_resize = false, crop_size = 20)
    @test exact(image).pixel_values ≈ exact(image ./ 255).pixel_values
end

@testset "settings" begin
    # the forms that transformers takes
    ip = CLIPImageProcessor(; size = 30, crop_size = 20, resample = :bilinear,
        image_mean = 0.5, image_std = [0.5, 0.5, 0.5])
    @test ip.size == (; shortest_edge = 30)
    @test ip.crop_size == (; height = 20, width = 20)
    @test ip.resample === 2 && ip.image_mean === 0.5 && ip.image_std === (0.5, 0.5, 0.5)
    @test CLIPImageProcessor(; size = 30, default_to_square = true).size ==
          (; height = 30, width = 30)
    @test size(CLIPImageProcessor(; size = (; height = 12, width = 16),
        do_center_crop = false)(rand(UInt8, 50, 30, 3)).pixel_values) == (16, 12, 3, 1)
    # overrides, which replace the integer sizes of the file
    proc = @test_logs create_processor(CLIP; crop_size = 200,
        size = (; height = 210, width = 220))
    @test proc.image_processor.crop_size == (; height = 200, width = 200)
    @test proc.image_processor.size == (; height = 210, width = 220)
    text = sprint(show, MIME("text/plain"), proc)
    @test occursin(
        r"crop_size = \(height = 200, width = 200\) +override \(preprocessor_config\.json: 224\)",
        text)
    @test_throws "`crop_size` must have `height` and `width` only" create_processor(CLIP;
        crop_size = (; height = 200))
    @test_throws "no component of CLIPProcessor has a setting `padding`" create_processor(
        CLIP; padding = true)
    @test_throws "(CLIPImageProcessor has `crop_size`)" create_processor(CLIP;
        crop_sise = 3)
    # others are errors that name the form of the setting, checked as the processor is made
    for (kwargs, message) in (
        (; do_center_crop = nothing) => "`do_center_crop` of CLIPImageProcessor is true",
        (; do_pad = true) => "`do_pad = true` is not supported by CLIPImageProcessor",
        (; resample = :lanczos) => "`resample = 1` (lanczos) is not supported",
        (; size = (; longest_edge = 30)) => "`size` must have `shortest_edge` only, or",
        (; size = (; shortest_edge = 0)) => "`size` must have `shortest_edge` only",
        (; crop_size = (20, 20)) => "`crop_size` must have `height` and `width` only",
        (; crop_size = (; height = 20, width = 2.5)) => "`crop_size` must have `height`",
        (; image_std = (0.5, 0.5)) => "`image_std` has one value per RGB channel")
        @test_throws message CLIPImageProcessor(; kwargs...)
    end
    @test_throws ["`size` must have `shortest_edge` only",
        "`size` is set in preprocessor_config.json"] mktempdir() do dir
        foreach(file -> cp(joinpath(CLIP, file), joinpath(dir, file)), readdir(CLIP))
        config = JSON.parsefile(joinpath(dir, "preprocessor_config.json"))
        write(joinpath(dir, "preprocessor_config.json"),
            JSON.json(merge(config, Dict("size" => Dict("shortest_edge" => 224,
                "longest_edge" => 448)))))
        create_processor(dir)
    end
    # the image processor's newer files name its class, which loads alike
    mktempdir() do dir
        foreach(file -> cp(joinpath(CLIP, file), joinpath(dir, file)), readdir(CLIP))
        config = JSON.parsefile(joinpath(dir, "preprocessor_config.json"))
        delete!(config, "feature_extractor_type")
        config["image_processor_type"] = "CLIPImageProcessorFast"
        config["processor_class"] = "CLIPProcessor"
        write(joinpath(dir, "preprocessor_config.json"), JSON.json(config))
        @test Metalhead.settings((@test_logs create_processor(dir)).image_processor) ==
              Metalhead.settings(create_processor(CLIP).image_processor)
    end
end

@testset "operations of the torchvision backend" begin
    # transformers' `get_resize_output_image_size`, rounded down
    @test T.get_resize_output_image_size(480, 640, 224) == (224, 298)
    @test T.get_resize_output_image_size(640, 480, 224) == (298, 224)
    @test T.get_resize_output_image_size(7, 7, 3) == (3, 3)
    # `center_crop`: offsets rounded down, zero padding with the extra row or column after
    x = reshape(1:(5 * 4), 5, 4, 1)
    @test T.center_crop(x, (2, 2))[:, :, 1] == x[2:3, 2:3, 1]
    padded = T.center_crop(x, (7, 6))
    @test size(padded) == (6, 7, 1)
    @test padded[1:5, 2:5, 1] == x[:, :, 1]
    @test all(iszero, padded[6, :, 1]) && all(iszero, padded[:, [1, 6, 7], 1])
end

# The tokenizer of the repository, HuggingFaceTokenizers' of its tokenizer.json, which
# METALHEAD_TEST_NETWORK adds to the test environment (runtests.jl)
const TEST_NETWORK = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))
if TEST_NETWORK
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers, Tokenizer
    @testset "HuggingFaceTokenizers" begin
        for name in CASES
            @testset "$name" begin
                check_outputs(name, run_case(name; network = true))
            end
        end
        tok = create_processor(CLIP_REPO).tokenizer
        @test tok isa Tokenizer
        for (text, ids) in TOKENIZER.ids
            @test Metalhead.encode_ids(tok, text) == ids
        end
        @test Metalhead.pad_token_id(tok) == TOKENIZER.pad
    end
end
