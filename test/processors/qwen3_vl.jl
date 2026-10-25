using Metalhead, Test
using JSON: JSON
using SafeTensors: SafeTensors
using Metalhead.Transformers: Qwen3VLProcessor, Qwen2VLImageProcessor, TokenizersBackend,
                              decode
const T = Metalhead.Transformers

include(joinpath(@__DIR__, "..", "processors_setup.jl"))

# Written by porting/dump_processor.py, through transformers' AutoProcessor
const QWEN_REPO = "Qwen/Qwen3.5-0.8B"
const QWEN = joinpath(pkgdir(Metalhead), "test", "fixtures", "processors",
    "Qwen--Qwen3.5-0.8B")
const FILE = joinpath(pkgdir(Metalhead), "test", "references", "processors",
    "qwen3_vl.safetensors")
const META = SafeTensors.deserialize(FILE).metadata
const REF = Metalhead.Checkpoint(FILE)
const CASES = sort!([k for (k, v) in META
                     if startswith(v, "{") && haskey(JSON.parse(v), "create")])

numbered(prefix) = [REF["$prefix/$i"] for i in 1:count(startswith("$prefix/"), keys(REF))]
settings(name) = JSON.parse(META[name])

# A backend that knows the prompts of the reference, with their placeholders expanded as
# transformers expands them: a tokenizer backend is any type with a method of `encode_ids`.
struct ReferenceBackend
    ids::Dict{String, Vector{Int}}
end
function ReferenceBackend()
    ids = Dict{String, Vector{Int}}()
    for name in CASES, (i, text) in enumerate(something(settings(name)["expanded"], []))
        tokens, mask = REF["$name/input_ids"][:, i], REF["$name/attention_mask"][:, i]
        ids[text] = tokens[mask .== 1]
    end
    return ReferenceBackend(ids)
end
function T.encode_ids(b::ReferenceBackend, text::AbstractString; add_special_tokens = true)
    return b.ids[text]
end

# The fixture's files with `legacy` added to its preprocessor_config.json (and the
# tokenizer.json of the repository, for strings).
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

# The outputs of the processor of a case, with its prompts as token ids (`strings = false`)
# or as strings, encoded by `tokenizer` (the reference backend, or tokenizer.json with
# `nothing`). Python's float images are already rescaled (`do_rescale = False`), which
# Metalhead's images of reals are.
function run_case(name; strings = false, tokenizer = ReferenceBackend(), kwargs...)
    m = settings(name)
    create = (; Metalhead.jsonvalue(m["create"])..., kwargs...)
    name == "float" && (create = Base.structdiff(create, (; do_rescale = false)))
    strings && tokenizer !== nothing && (create = (; create..., tokenizer))
    ids = numbered("$name/text_ids")
    text = m["text"] === nothing ? nothing : strings ? m["text"] :
           m["text"] isa String ? only(ids) : ids
    images = numbered("$name/image")
    call = Metalhead.jsonvalue(something(m["call"], Dict()))
    process(dir) = create_processor(dir; create...)(;
        images = isempty(images) ? nothing : images, text, call...)
    tokenizer_json = strings && tokenizer === nothing
    m["legacy"] === nothing && return process(tokenizer_json ? QWEN_REPO : QWEN)
    return with_legacy(process, m["legacy"]; tokenizer_json)
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
    @test proc.tokenizer isa TokenizersBackend
    # preprocessor_config.json over the defaults of the Python class
    @test ip.patch_size == 16 && ip.merge_size == 2 && ip.temporal_patch_size == 2
    @test ip.size == (; longest_edge = 16_777_216, shortest_edge = 65_536)
    @test ip.image_mean == ip.image_std == (0.5, 0.5, 0.5)
    @test ip.resample == 3 && ip.do_resize && ip.do_convert_rgb
    default = Qwen2VLImageProcessor()
    @test default.patch_size == 14
    @test default.image_mean == (0.48145466, 0.4578275, 0.40821073)
    @test default.size == (; shortest_edge = 3136, longest_edge = 1_003_520)
    tok = proc.tokenizer
    @test tok.model_specific_special_tokens.image_token == "<|image_pad|>"
    @test tok.added_tokens_decoder[248_056] == "<|image_pad|>"
    @test tok.pad_token == "<|endoftext|>" && tok.padding_side == "right"
    @test all(settings(name)["cls"] == "Qwen3VLProcessor" for name in CASES)
    text = sprint(show, MIME("text/plain"), proc)
    @test occursin(r"patch_size = 16 +preprocessor_config\.json", text)
    @test occursin(r"resample = 3 \(bicubic\) +class default", text)
    @test startswith(sprint(show, proc),
        "Qwen3VLProcessor(image_processor = Qwen2VLImageProcessor(do_resize = true, ")
    @test endswith(sprint(show, proc), "init_kwargs = NamedTuple()))")
    # by the model type of config.json, without a `processor_class`
    mktempdir() do dir
        foreach(file -> cp(joinpath(QWEN, file), joinpath(dir, file)), readdir(QWEN))
        config = JSON.parsefile(joinpath(dir, "preprocessor_config.json"))
        delete!(config, "processor_class")
        write(joinpath(dir, "preprocessor_config.json"), JSON.json(config))
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

@testset "strings" begin
    # the placeholders of strings are expanded before tokenizing, as in transformers: the
    # reference backend knows the expanded prompts only
    for name in CASES
        settings(name)["text"] === nothing && continue
        @testset "$name" begin
            check_outputs(name, run_case(name; strings = true))
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
    proc = create_processor(QWEN; patch_size = 4,
        size = (; shortest_edge = 64, longest_edge = 4096))
    image = only(numbered("exact/image"))  # 6 image tokens
    ids = only(numbered("exact/text_ids"))
    pad = 248_056
    @test count(==(pad), ids) == 1
    one = proc(; images = image, text = ids)
    @test one.input_ids isa Vector{Int} && one.mm_token_type_ids isa Vector{Int}
    @test count(==(pad + 1), one.input_ids) == 6
    @test one.mm_token_type_ids == Int.(one.input_ids .== pad + 1)
    @test keys(one) == (:input_ids, :attention_mask, :mm_token_type_ids, :pixel_values,
        :image_grid_thw)
    batch = proc(; images = [image], text = [ids])
    @test batch.input_ids == reshape(one.input_ids, :, 1)
    # the image processor alone
    ip = proc.image_processor
    @test ip(image) == (; one.pixel_values, one.image_grid_thw)
    @test ip(cat(image, image; dims = 4)).image_grid_thw == [1 1; 4 4; 6 6]
    # as many placeholders as images
    @test_throws "2 image placeholders (`<|image_pad|>`) for 1 images" proc(;
        images = image, text = [ids; ids])
    @test_throws "1 image placeholders (`<|image_pad|>`) for 2 images" proc(;
        images = [image, image], text = ids)
    @test_throws "0 image placeholders" proc(; images = image, text = [1, 2, 3])
    # without images, the placeholders are left as they are
    @test proc(; text = ids).input_ids == ids .+ 1
    @test_throws "`images` or `text`" proc()
    # strings, and truncation that cuts image tokens
    strings = create_processor(QWEN; patch_size = 4, tokenizer = ReferenceBackend(),
        size = (; shortest_edge = 64, longest_edge = 4096))
    prompt = settings("exact")["text"]
    @test strings(; images = image, text = prompt) == one
    @test_throws "0 image placeholders" strings(; images = image, text = "No image")
    @test_throws "2 image placeholders" strings(; images = image, text = prompt^2)
    @test_throws "image tokens" strings(; images = image, text = prompt, truncation = true,
        max_length = 5)
    @test_throws "longer than `max_length = 5`" proc(; images = image, text = ids,
        max_length = 5)
    # a processor without a tokenizer takes images only, and one without an image
    # processor text only
    @test Qwen3VLProcessor(ip, nothing)(; images = image) == ip(image)
    @test_throws "no tokenizer" Qwen3VLProcessor(ip, nothing)(; text = ids)
    text_only = Qwen3VLProcessor(nothing, proc.tokenizer)
    @test text_only(; text = ids) == proc(; text = ids)
    @test_throws "Qwen3VLProcessor has no image processor for `images`" text_only(;
        images = image, text = ids)
    @test_throws "the `image_processor` of Qwen3VLProcessor is a Qwen2VLImageProcessor" Qwen3VLProcessor(
        proc.tokenizer, nothing)
    # the keywords of the tokenizer
    @test proc(; text = [ids, ids[1:3]], padding_side = "left").attention_mask[1, 2] == 0
    @test_throws "Qwen3VLProcessor has no keyword `return_tensors`" proc(; text = ids,
        return_tensors = "pt")
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
    swapped = Qwen3VLProcessor(ip, proc.tokenizer)
    out = swapped(; images = numbered("exact/image"), text = only(numbered("exact/text_ids")))
    @test out.input_ids == vec(REF["exact/input_ids"]) .+ 1
end

# Strings through HuggingFaceTokenizers.jl, which METALHEAD_TEST_NETWORK adds to the test
# environment (runtests.jl), with the tokenizer.json of the repository
const TEST_NETWORK = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))
if TEST_NETWORK
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers
    @testset "HuggingFaceTokenizers" begin
        proc = @test_logs create_processor(QWEN_REPO)
        @test Metalhead.settings(proc.image_processor) ==
              Metalhead.settings(create_processor(QWEN).image_processor)
        for name in CASES
            settings(name)["text"] === nothing && continue
            @testset "$name" begin
                check_outputs(name, run_case(name; strings = true, tokenizer = nothing))
            end
        end
        # the tokenizer of transformers (Qwen3_5Tokenizer) rebuilds the normaliser and the
        # pre-tokenizer of tokenizer.json in Python: the same ids and texts
        @test META["tokenizer/class"] == "Qwen3_5Tokenizer"
        tok = proc.tokenizer
        decoded = JSON.parse(META["tokenizer/decoded"])
        for (i, text) in enumerate(JSON.parse(META["tokenizer/texts"]))
            ids = REF["tokenizer/ids/$i"]
            @test tok(text).input_ids == ids .+ 1
            for skip in (false, true)
                @test decode(tok, ids .+ 1; skip_special_tokens = skip) ==
                      decoded[i][skip ? "True" : "False"]
            end
        end
        # the example of the docstring
        prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" *
                 "Describe this image.<|im_end|>\n<|im_start|>assistant\n" *
                 "<think>\n\n</think>\n\n"
        inputs = proc(; images = rand(UInt8, 640, 480, 3), text = prompt)
        @test count(==(248_056 + 1), inputs.input_ids) == 300
        @test decode(proc, inputs.input_ids[1:3]) == "<|im_start|>user\n"
    end
end
