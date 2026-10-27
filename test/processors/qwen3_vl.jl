using Metalhead, Test
using JSON: JSON
using SafeTensors: SafeTensors
using Metalhead.Transformers: Qwen3VLProcessor, Qwen2VLImageProcessor
const T = Metalhead.Transformers

include(joinpath(@__DIR__, "..", "processors_setup.jl"))
include(joinpath(@__DIR__, "..", "data.jl"))
using .TestData: test_data

# Written by porting/dump_processor.py, through transformers' AutoProcessor
const QWEN_REPO = "Qwen/Qwen3.5-0.8B"
const QWEN = joinpath(pkgdir(Metalhead), "test", "fixtures", "processors",
    "Qwen--Qwen3.5-0.8B")
const FILE = test_data("references/processors/qwen3_vl.safetensors")
const META = SafeTensors.deserialize(FILE).metadata
const REF = Metalhead.Checkpoint(FILE)
const CASES = sort!([k for (k, v) in META
                     if startswith(v, "{") && haskey(JSON.parse(v), "create")])

numbered(prefix) = [REF["$prefix/$i"] for i in 1:count(startswith("$prefix/"), keys(REF))]
settings(name) = JSON.parse(META[name])

# The ids of the image and video placeholders and of the padding token of the tokenizer
const IMAGE_ID = 248_056
const VIDEO_ID = only(unique(REF["text_only/input_ids"][REF["text_only/mm_token_type_ids"] .== 2]))
const PAD_ID = only(unique(REF["batch/input_ids"][REF["batch/attention_mask"] .== 0]))

# A stand-in for the tokenizer of Qwen3.5, offline: the ids of the prompts of the cases,
# with their placeholders expanded as transformers expands them
struct ReferenceTokenizer
    ids::Dict{String, Vector{Int}}
    pad::Union{Nothing, Int}
end
function ReferenceTokenizer()
    ids = Dict{String, Vector{Int}}()
    for name in CASES, (i, text) in enumerate(something(settings(name)["expanded"], []))
        tokens, mask = REF["$name/input_ids"][:, i], REF["$name/attention_mask"][:, i]
        ids[text] = tokens[mask .== 1]
    end
    return ReferenceTokenizer(ids, PAD_ID)
end
function Metalhead.encode_ids(tok::ReferenceTokenizer, text::AbstractString;
        add_special_tokens::Bool = true)
    return tok.ids[text]
end
function Metalhead.token_id(::ReferenceTokenizer, token::AbstractString)
    return get(Dict("<|image_pad|>" => IMAGE_ID, "<|video_pad|>" => VIDEO_ID), token, nothing)
end
Metalhead.pad_token_id(tok::ReferenceTokenizer) = tok.pad

const TOKENIZER = ReferenceTokenizer()

# `proc` with the tokenizer `tok`
with_tokenizer(proc, tok = TOKENIZER) = Qwen3VLProcessor(proc.image_processor, tok;
    proc.provenance)

# The fixture's files with `legacy` added to its preprocessor_config.json (and the
# tokenizer.json of the repository, with `tokenizer_json`).
function with_legacy(f, legacy; tokenizer_json = false)
    mktempdir() do dir
        foreach(file -> cp(joinpath(QWEN, file), joinpath(dir, file)), readdir(QWEN))
        tokenizer_json && cp(Metalhead.hf_download(QWEN_REPO, "tokenizer.json";
                revision = META["revision"]), joinpath(dir, "tokenizer.json");
            follow_symlinks = true)
        config = JSON.parsefile(joinpath(dir, "preprocessor_config.json"))
        write(joinpath(dir, "preprocessor_config.json"), JSON.json(merge(config, legacy)))
        return f(dir)
    end
end

# The outputs of the processor of a case: offline, the processor of the fixture files with
# the stand-in for the tokenizer; with `network`, that of the repository, whose tokenizer is
# HuggingFaceTokenizers' of its tokenizer.json. The only keyword of the calls of the cases,
# the `padding = True` of "batch", is how the processor pads a batch. Python's float images
# are already rescaled (`do_rescale = False`), which Metalhead's images of reals are.
function run_case(name; network = false)
    m = settings(name)
    create = Metalhead.jsonvalue(m["create"])
    name == "float" && (create = Base.structdiff(create, (; do_rescale = false)))
    images = numbered("$name/image")
    function process(dir)
        proc = create_processor(dir; create...)
        network || (proc = with_tokenizer(proc))
        return proc(; images = isempty(images) ? nothing : images, text = m["text"])
    end
    m["legacy"] === nothing && return process(network ? QWEN_REPO : QWEN)
    return with_legacy(process, m["legacy"]; tokenizer_json = network)
end

# Whether the outputs of the processor are transformers': exact, but for images of reals.
function check_outputs(name, out)
    m = settings(name)
    exact = !(name in ("float", "float_no_rescale"))
    expected = Set(split(k, '/')[2] for k in keys(REF)
                   if startswith(k, "$name/") && !occursin(r"/(image|text_ids)/", k))
    @test Set(string.(keys(out))) == expected
    for k in keys(out)
        r = REF["$name/$k"]
        x = out[k]
        x isa AbstractVector && (r = vec(r))  # one text
        if k === :pixel_values
            m["stride"] === nothing || (x = x[:, 1:m["stride"]:end])
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
    proc = @test_logs create_processor(QWEN)
    @test proc isa Qwen3VLProcessor
    ip = proc.image_processor
    @test ip isa Qwen2VLImageProcessor
    @test proc.tokenizer === nothing  # the fixture has no tokenizer.json
    # preprocessor_config.json over the defaults of the Python class
    @test ip.patch_size == 16 && ip.merge_size == 2 && ip.temporal_patch_size == 2
    @test ip.size == (; longest_edge = 16_777_216, shortest_edge = 65_536)
    @test ip.image_mean == ip.image_std == (0.5, 0.5, 0.5)
    @test ip.resample == 3 && ip.do_resize && ip.do_convert_rgb
    default = Qwen2VLImageProcessor()
    @test default.patch_size == 14
    @test default.image_mean == (0.48145466, 0.4578275, 0.40821073)
    @test default.size == (; shortest_edge = 3136, longest_edge = 1_003_520)
    @test all(settings(name)["cls"] == "Qwen3VLProcessor" for name in CASES)
    text = sprint(show, MIME("text/plain"), proc)
    @test occursin(r"patch_size = 16 +preprocessor_config\.json", text)
    @test occursin(r"resample = 3 \(bicubic\) +class default", text)
    @test startswith(sprint(show, proc),
        "Qwen3VLProcessor(image_processor = Qwen2VLImageProcessor(do_resize = true, ")
    @test endswith(sprint(show, proc), "), tokenizer = nothing)")
    # by the model type of config.json, without a `processor_class`
    mktempdir() do dir
        foreach(file -> cp(joinpath(QWEN, file), joinpath(dir, file)), readdir(QWEN))
        config = JSON.parsefile(joinpath(dir, "preprocessor_config.json"))
        delete!(config, "processor_class")
        write(joinpath(dir, "preprocessor_config.json"), JSON.json(config))
        @test create_processor(dir) isa Qwen3VLProcessor
        # and of Qwen3.5 MoE, whose repositories (Qwen3.6-35B-A3B) have these files
        config = JSON.parsefile(joinpath(dir, "config.json"))
        config["model_type"] = "qwen3_5_moe"
        write(joinpath(dir, "config.json"), JSON.json(config))
        @test create_processor(dir) isa Qwen3VLProcessor
    end
end

@testset "transformers' processor" begin
    @test length(CASES) == 19
    for name in CASES
        @testset "$name" begin
            check_outputs(name, run_case(name))
        end
    end
end

@testset "documented outputs" begin
    outputs = [run_case(name) for name in CASES]
    check_documented_keys(Qwen3VLProcessor, outputs)
    images = [o for o in outputs if !haskey(o, :input_ids)]
    @test length(images) == 1
    check_documented_keys(Qwen2VLImageProcessor, images)
    proc = create_processor(QWEN; patch_size = 4)
    check_documented_keys(Qwen2VLImageProcessor,
        [proc.image_processor(numbered("batch/image"))])
end

@testset "placeholders" begin
    proc = with_tokenizer(create_processor(QWEN; patch_size = 4,
        size = (; shortest_edge = 64, longest_edge = 4096)))
    image = only(numbered("exact/image"))  # 6 image tokens
    prompt = settings("exact")["text"]
    @test count("<|image_pad|>", prompt) == 1
    one = proc(; images = image, text = prompt)
    @test one.input_ids isa Vector{Int} && one.mm_token_type_ids isa Vector{Int}
    @test count(==(IMAGE_ID + 1), one.input_ids) == 6
    @test one.mm_token_type_ids == Int.(one.input_ids .== IMAGE_ID + 1)
    @test keys(one) == (:input_ids, :attention_mask, :mm_token_type_ids, :pixel_values,
        :image_grid_thw)
    batch = proc(; images = [image], text = [prompt])
    @test batch.input_ids == reshape(one.input_ids, :, 1)
    # the image processor alone
    ip = proc.image_processor
    @test ip(image) == (; one.pixel_values, one.image_grid_thw)
    @test ip(cat(image, image; dims = 4)).image_grid_thw == [1 1; 4 4; 6 6]
    # as many placeholders as images
    @test_throws "2 image placeholders (`<|image_pad|>`) for 1 images" proc(;
        images = image, text = prompt^2)
    @test_throws "1 image placeholders (`<|image_pad|>`) for 2 images" proc(;
        images = [image, image], text = prompt)
    @test_throws "0 image placeholders" proc(; images = image, text = "No image")
    # without images, the placeholders are left as they are
    text = settings("text_only")["text"]
    @test proc(; text).input_ids == vec(REF["text_only/input_ids"]) .+ 1
    @test_throws "`images` or `text`" proc()
    # strings only: token ids aren't text, as in transformers
    ids = only(numbered("exact/text_ids"))
    @test_throws "`text` is a string or a vector of strings" proc(; images = image,
        text = ids)
    @test_throws "`text` is a string or a vector of strings" proc(; text = [ids])
    # a batch needs a padding token
    nopad = Qwen3VLProcessor(nothing, ReferenceTokenizer(TOKENIZER.ids, nothing))
    @test_throws "the tokenizer has no padding token" nopad(;
        text = settings("batch")["expanded"])
    # a processor without a tokenizer takes images only, and one without an image
    # processor text only
    @test Qwen3VLProcessor(ip, nothing)(; images = image) == ip(image)
    @test_throws ["Qwen3VLProcessor has no tokenizer to encode text",
        "`using HuggingFaceTokenizers` before `create_processor`"] Qwen3VLProcessor(ip,
        nothing)(; text = prompt)
    text_only = Qwen3VLProcessor(nothing, proc.tokenizer)
    @test text_only(; text) == proc(; text)
    @test_throws "Qwen3VLProcessor has no image processor for `images`" text_only(;
        images = image, text = prompt)
    @test_throws "the `image_processor` of Qwen3VLProcessor is a Qwen2VLImageProcessor" Qwen3VLProcessor(
        proc.tokenizer, nothing)
end

@testset "image processor" begin
    # the sizes that transformers' `smart_resize` gives with the settings of Qwen3.5
    s = JSON.parse(META["smart_resize"])
    for (height, width, h, w) in eachcol(REF["smart_resize"])
        @test T.smart_resize(height, width; factor = s["factor"],
            min_pixels = s["min_pixels"], max_pixels = s["max_pixels"]) == (h, w)
    end
    @test_throws "aspect ratio" T.smart_resize(10, 2001)
    ip = create_processor(QWEN).image_processor
    @test ip(rand(UInt8, 640, 480, 3)).image_grid_thw == [1; 30; 40;;]
    @test size(ip(rand(UInt8, 640, 480, 3)).pixel_values) == (1536, 1200)

    # the patches, merge window by merge window, each `(patch, patch, frame, channel)`
    image = [w + 10h + 100c for w in 1:8, h in 1:12, c in 1:3] ./ 1000
    ip = Qwen2VLImageProcessor(; patch_size = 2, temporal_patch_size = 3,
        do_normalize = false, size = (; shortest_edge = 1, longest_edge = 4096))
    out = ip(image)
    @test out.image_grid_thw == [1; 6; 4;;]
    patches = reshape(out.pixel_values, 2, 2, 3, 3, :)
    windows = [(br, bc, ir, ic) for br in 0:2 for bc in 0:1 for ir in 0:1 for ic in 0:1]
    @test all(enumerate(windows)) do (n, (br, bc, ir, ic))
        row, col = 2br + ir, 2bc + ic
        all(patches[pw, ph, t, c, n] ≈ image[2col + pw, 2row + ph, c]
        for pw in 1:2, ph in 1:2, t in 1:3, c in 1:3)
    end
    # 8-bit images and images of reals, of different sizes
    ip = Qwen2VLImageProcessor(; patch_size = 4, size = (; shortest_edge = 1,
        longest_edge = 1 << 20))
    image = rand(UInt8, 64, 96, 3)
    @test ip(image).pixel_values ≈ ip(image ./ 255).pixel_values
    out = ip([rand(UInt8, 30, 20, 3), rand(Float32, 16, 16, 3)])
    @test out.image_grid_thw == [1 1; 4 4; 8 4]  # 30 × 20 resized to 32 × 16
    @test size(out.pixel_values) == (4 * 4 * 2 * 3, 32 + 16)
    @test_throws "channels" Qwen2VLImageProcessor(; do_convert_rgb = false)(rand(UInt8,
        32, 32, 1))
    @test_throws "multiples of `patch_size * merge_size` (28), got 30 × 28" Qwen2VLImageProcessor(;
        do_resize = false)(rand(UInt8, 30, 28, 3))
    @test_throws "no images" ip([])
end

@testset "settings" begin
    # overrides, merged into `size`; the legacy `min_pixels` and `max_pixels`
    proc = @test_logs create_processor(QWEN; size = (; longest_edge = 1 << 20))
    @test proc.image_processor.size == (; longest_edge = 1 << 20, shortest_edge = 65_536)
    @test occursin(r"size = \(longest_edge = 1048576, shortest_edge = 65536\) +override \(preprocessor_config\.json: \(longest_edge = 16777216, shortest_edge = 65536\)\)",
        sprint(show, MIME("text/plain"), proc))
    proc = create_processor(QWEN; min_pixels = 1024)
    @test proc.image_processor.size == (; longest_edge = 16_777_216, shortest_edge = 1024)
    with_legacy(Dict("min_pixels" => 128, "max_pixels" => 1024)) do dir
        proc = @test_logs create_processor(dir)
        @test proc.image_processor.size == (; longest_edge = 1024, shortest_edge = 128)
        @test occursin(r"size = .* +preprocessor_config\.json \(`m.._pixels`, `m.._pixels`\)",
            sprint(show, MIME("text/plain"), proc))
    end
    # a `size` of transformers 4.45, whose entries the legacy keys replace
    old = Dict("size" => Dict("min_pixels" => 3136, "max_pixels" => 12_845_056),
        "min_pixels" => 3136, "max_pixels" => 12_845_056)
    with_legacy(old) do dir
        size = create_processor(dir).image_processor.size
        @test Dict(pairs(size)) ==
              Dict(:shortest_edge => 3136, :longest_edge => 12_845_056)
    end
    with_legacy(Dict("size" => Dict("min_pixels" => 3136))) do dir
        @test_throws "`size` must have positive `shortest_edge` and `longest_edge` only" create_processor(dir)
    end
    # Qwen2-VL's configuration, without `size`, over the class defaults
    qwen2_vl = Dict("min_pixels" => 3136, "max_pixels" => 12_845_056, "patch_size" => 14,
        "temporal_patch_size" => 2, "merge_size" => 2,
        "image_mean" => [0.48145466, 0.4578275, 0.40821073],
        "image_std" => [0.26862954, 0.26130258, 0.27577711],
        "image_processor_type" => "Qwen2VLImageProcessor")
    mktempdir() do dir
        write(joinpath(dir, "preprocessor_config.json"), JSON.json(qwen2_vl))
        ip = @test_logs create_processor(dir)
        @test ip isa Qwen2VLImageProcessor
        @test ip.size == (; shortest_edge = 3136, longest_edge = 12_845_056)
    end
    # ignored, unknown and unsupported settings
    with_legacy(Dict("do_center_crop" => true, "crop_size" => Dict("height" => 2))) do dir
        @test_logs create_processor(dir)
    end
    with_legacy(Dict("qwen_test_key" => 1)) do dir
        @test_logs (:warn,
            "`qwen_test_key` in preprocessor_config.json is not a setting of Qwen2VLImageProcessor; it is ignored") create_processor(dir)
    end
    @test_throws "no component of Qwen3VLProcessor has a setting `patch_sise` (Qwen2VLImageProcessor has `patch_size`)" create_processor(QWEN;
        patch_sise = 14)
    @test_throws "`resample = 1` (lanczos) is not supported" create_processor(QWEN;
        resample = 1)
    @test_throws "`merge_size` must be positive" create_processor(QWEN; merge_size = 0)
    @test_throws "`size` must have" create_processor(QWEN; size = (; height = 4))
    @test_throws "`chat_template` is not used by" create_processor(QWEN;
        chat_template = "")
    # by hand, over the class defaults, and with another image processor
    proc = create_processor(QWEN)
    ip = Qwen2VLImageProcessor(; patch_size = 4, image_mean = (0.5, 0.5, 0.5),
        image_std = (0.5, 0.5, 0.5), size = (; shortest_edge = 64, longest_edge = 4096))
    swapped = Qwen3VLProcessor(ip, TOKENIZER)
    out = swapped(; images = numbered("exact/image"), text = settings("exact")["text"])
    @test out.input_ids == vec(REF["exact/input_ids"]) .+ 1
end

# The tokenizer of the repository, HuggingFaceTokenizers' of its tokenizer.json, which
# METALHEAD_TEST_NETWORK adds to the test environment (runtests.jl)
const TEST_NETWORK = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))
if TEST_NETWORK
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers, Tokenizer
    @testset "HuggingFaceTokenizers" begin
        proc = @test_logs create_processor(QWEN_REPO)
        @test proc.tokenizer isa Tokenizer
        @test Metalhead.settings(proc.image_processor) ==
              Metalhead.settings(create_processor(QWEN).image_processor)
        tok = proc.tokenizer
        # transformers' tokenizer (Qwen3_5Tokenizer) also has the tokens of
        # tokenizer_config.json that tokenizer.json lacks (`<|audio_pad|>`, `<tts_pad>`),
        # which HuggingFaceTokenizers' tokenizer of tokenizer.json doesn't: texts with them
        # differ
        added = JSON.parsefile(joinpath(QWEN, "tokenizer_config.json"))["added_tokens_decoder"]
        lacking = [t["content"] for t in values(added)
                   if Metalhead.token_id(tok, t["content"]) === nothing]
        @test "<|audio_pad|>" in lacking && !("<|image_pad|>" in lacking)
        with_lacking(text) = any(t -> occursin(t, text), lacking)
        for name in CASES
            text = settings(name)["text"]
            (text === nothing || any(with_lacking, vcat(text))) && continue
            @testset "$name" begin
                check_outputs(name, run_case(name; network = true))
            end
        end
        # the ids of the stand-in are the tokenizer's
        @test Metalhead.token_id(tok, "<|image_pad|>") == IMAGE_ID
        @test Metalhead.token_id(tok, "<|video_pad|>") == VIDEO_ID
        @test Metalhead.pad_token_id(tok) == PAD_ID
        # transformers' tokenizer (Qwen3_5Tokenizer) rebuilds the normaliser and the
        # pre-tokenizer of tokenizer.json in Python: the same ids and texts
        @test META["tokenizer/class"] == "Qwen3_5Tokenizer"
        decoded = JSON.parse(META["tokenizer/decoded"])
        for (i, text) in enumerate(JSON.parse(META["tokenizer/texts"]))
            with_lacking(text) && continue
            ids = REF["tokenizer/ids/$i"]
            @test Metalhead.encode_ids(tok, text) == ids
            for skip in (false, true)
                @test HuggingFaceTokenizers.decode(tok, ids; skip_special_tokens = skip) ==
                      decoded[i][skip ? "True" : "False"]
            end
        end
        # the example of the docstring
        prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" *
                 "Describe this image.<|im_end|>\n<|im_start|>assistant\n" *
                 "<think>\n\n</think>\n\n"
        inputs = proc(; images = rand(UInt8, 640, 480, 3), text = prompt)
        @test count(==(IMAGE_ID + 1), inputs.input_ids) == 300
        @test HuggingFaceTokenizers.decode(tok, inputs.input_ids[1:3] .- 1;
            skip_special_tokens = false) == "<|im_start|>user\n"
    end

    # Qwen3.8 has the preprocessor_config.json of Qwen3.5, and a tokenizer.json with the
    # tokens that Qwen3.5's lacks: the inputs that transformers' processor made of the
    # prompts and the image of the reference of its model (test/models/qwen3_5.jl)
    @testset "Qwen/Qwen3.8-27B" begin
        repo = "Qwen/Qwen3.8-27B"
        file = test_data("references/Qwen--Qwen3.8-27B.safetensors")
        meta = SafeTensors.deserialize(file).metadata
        ref = Metalhead.Checkpoint(file)
        proc = @test_logs create_processor(repo)
        @test proc isa Qwen3VLProcessor && proc.tokenizer isa Tokenizer
        @test Metalhead.settings(proc.image_processor) ==
              Metalhead.settings(create_processor(QWEN).image_processor)
        @test Metalhead.token_id(proc.tokenizer, "<|audio_pad|>") !== nothing
        ids = ref["input_ids"]
        for (b, prompt) in enumerate(JSON.parse(meta["prompts"]))
            @test Metalhead.encode_ids(proc.tokenizer, prompt)[1:size(ids, 1)] == ids[:, b]
        end
        # transformers' `min_pixels` kept the size of the image
        image = ref["image/image"]
        proc = create_processor(repo;
            size = (; shortest_edge = size(image, 1) * size(image, 2),
                longest_edge = 16_777_216))
        out = proc(; images = image, text = meta["image_text"])
        @test out.input_ids == vec(ref["image/input_ids"]) .+ 1
        @test out.image_grid_thw == ref["image/image_grid_thw"]
        @test Float16.(out.pixel_values) == ref["image/pixel_values"]
    end
end
