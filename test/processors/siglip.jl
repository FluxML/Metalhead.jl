using Metalhead, Test
using JSON: JSON
using SafeTensors: SafeTensors
using Metalhead.Transformers: SiglipProcessor, SiglipImageProcessor, Siglip2Processor,
                              Siglip2ImageProcessor, image_size_for_max_num_patches

include(joinpath(@__DIR__, "..", "processors_setup.jl"))

# Written by porting/dump_processor.py, through transformers' AutoProcessor
const PROCESSORS = joinpath(pkgdir(Metalhead), "test", "fixtures", "processors")
const REFERENCES = joinpath(pkgdir(Metalhead), "test", "references", "processors")
const SIGLIP = joinpath(PROCESSORS, "google--siglip-base-patch16-224")
const SIGLIP2_224 = joinpath(PROCESSORS, "google--siglip2-base-patch16-224")
const NAFLEX = joinpath(PROCESSORS, "google--siglip2-base-patch16-naflex")

struct References
    meta::Dict{String, String}
    ref::Metalhead.Checkpoint
    cases::Vector{String}
end
function References(file)
    meta = SafeTensors.deserialize(file).metadata
    cases = sort!([k for (k, v) in meta if startswith(v, "{") && haskey(JSON.parse(v), "create")])
    return References(meta, Metalhead.Checkpoint(file), cases)
end
const SIGLIP_REF = References(joinpath(REFERENCES, "siglip.safetensors"))
const NAFLEX_REF = References(joinpath(REFERENCES, "siglip2.safetensors"))

numbered(r, prefix) = [r.ref["$prefix/$i"]
                       for i in 1:count(k -> occursin(Regex("^$prefix/\\d+\$"), k), keys(r.ref))]
case_texts(m) = m["text"] === nothing ? String[] :
                m["text"] isa AbstractString ? [m["text"]] : Vector{String}(m["text"])

# A stand-in for the tokenizers of the repositories, offline: the ids that transformers'
# tokenizer gives the prompts of the cases, under the text that Metalhead's processor
# encodes (lowercased by Siglip2Processor), and the padding token of the tokenizer
struct ReferenceTokenizer
    ids::Dict{String, Vector{Int}}
    pad::Int
end
function Metalhead.encode_ids(tok::ReferenceTokenizer, text::AbstractString;
        add_special_tokens::Bool = true)
    return tok.ids[text]
end
Metalhead.pad_token_id(tok::ReferenceTokenizer) = tok.pad

function reference_ids(r::References, repo; lower = false)
    ids = Dict{String, Vector{Int}}()
    for name in r.cases
        m = JSON.parse(r.meta[name])
        m["repo"] == repo || continue
        for (i, t) in enumerate(case_texts(m))
            ids[lower ? lowercase(t) : t] = r.ref["$name/text_ids/$i"]
        end
    end
    return ids
end
# the padding tokens `</s>` of SigLIP's tokenizer and `<pad>` of Gemma's (SigLIP 2)
const SIGLIP_TOKENIZER = ReferenceTokenizer(
    reference_ids(SIGLIP_REF, "google/siglip-base-patch16-224"), 1)
const GEMMA_TOKENIZER = ReferenceTokenizer(
    reference_ids(SIGLIP_REF, "google/siglip2-base-patch16-224"), 0)
const NAFLEX_TOKENIZER = ReferenceTokenizer(
    reference_ids(NAFLEX_REF, "google/siglip2-base-patch16-naflex"; lower = true), 0)

with_tokenizer(proc, tok) = typeof(proc)(proc.image_processor, tok;
    Metalhead.settings(proc)..., proc.provenance)

# The processor and the inputs of a case: offline, the processor of the fixture files with
# the stand-in for the tokenizer; with `network`, that of the repository. Python's float
# images are already rescaled (`do_rescale = False`), which Metalhead's images of reals are.
function case(r::References, name; network = false)
    m = JSON.parse(r.meta[name])
    create = Metalhead.jsonvalue(m["create"])
    endswith(name, "float") && (create = Base.structdiff(create, (; do_rescale = false)))
    proc = if network
        create_processor(m["repo"]; revision = m["revision"], create...)
    else
        dir = joinpath(PROCESSORS, replace(m["repo"], "/" => "--"))
        tok = m["repo"] == "google/siglip-base-patch16-224" ? SIGLIP_TOKENIZER :
              m["repo"] == "google/siglip2-base-patch16-224" ? GEMMA_TOKENIZER :
              NAFLEX_TOKENIZER
        with_tokenizer(create_processor(dir; create...), tok)
    end
    images = numbered(r, "$name/image")
    call = m["call"] === nothing ? (;) : Metalhead.jsonvalue(m["call"])
    return proc, (; images = isempty(images) ? nothing : images, text = m["text"], call...)
end

# Whether the outputs of the processor are transformers': exact, but for images of reals.
function check_outputs(r::References, name, out)
    exact = !(endswith(name, "float") || endswith(name, "float_no_rescale"))
    expected = Set(split(k, '/')[end] for k in keys(r.ref)
                   if startswith(k, "$name/") && count('/', chopprefix(k, "$name/")) == 0)
    @test Set(string.(keys(out))) == expected
    stride = get(JSON.parse(r.meta[name]), "stride", 1)
    for k in keys(out)
        ref = r.ref["$name/$k"]
        if k === :input_ids
            @test out[k] == ref .+ 1
        elseif k === :pixel_values
            x = ndims(out[k]) == 3 ? out[k][:, 1:stride:end, :] : out[k]
            @test eltype(x) == Float32
            @test size(x) == size(ref)
            @test exact ? x == ref : maximum(abs.(x .- ref)) <= 1.0e-6 * maximum(abs, ref)
        else
            @test out[k] == ref
        end
    end
end

@testset "loading" begin
    proc = @test_logs create_processor(SIGLIP)
    @test proc isa SiglipProcessor && proc.image_processor isa SiglipImageProcessor
    @test proc.tokenizer === nothing  # the fixture has no tokenizer.json
    ip = proc.image_processor
    @test ip.size == (; height = 224, width = 224) && ip.resample == 3
    @test ip.image_mean == ip.image_std == (0.5, 0.5, 0.5) && ip.do_convert_rgb
    # a SigLIP 2 checkpoint of fixed resolution, whose config.json names no architecture
    gemma = @test_logs create_processor(SIGLIP2_224)
    @test gemma isa SiglipProcessor && gemma.image_processor.resample == 2
    @test gemma.image_processor.do_convert_rgb  # `null` in the file: the class default
    naflex = @test_logs create_processor(NAFLEX)
    @test naflex isa Siglip2Processor && naflex.image_processor isa Siglip2ImageProcessor
    nip = naflex.image_processor
    @test nip.patch_size == 16 && nip.max_num_patches == 256 && nip.resample == 2
    @test !nip.do_convert_rgb
end

@testset "transformers' processor" begin
    @test length(SIGLIP_REF.cases) == 29 && length(NAFLEX_REF.cases) == 19
    for r in (SIGLIP_REF, NAFLEX_REF), name in r.cases
        proc, inputs = case(r, name)
        @testset "$name" begin
            check_outputs(r, name, proc(; inputs...))
        end
    end
    # the image processors alone
    proc, inputs = case(SIGLIP_REF, "batch")
    @test proc.image_processor(inputs.images) ==
          (; pixel_values = SIGLIP_REF.ref["batch/pixel_values"])
    proc, inputs = case(NAFLEX_REF, "batch")
    out = proc.image_processor(inputs.images)
    @test keys(out) == (:pixel_values, :pixel_attention_mask, :spatial_shapes)
    @test out.pixel_values == NAFLEX_REF.ref["batch/pixel_values"]
    @test proc.image_processor(inputs.images[2]).spatial_shapes ==
          NAFLEX_REF.ref["batch/spatial_shapes"][:, 2:2]
    # transformers' `get_image_size_for_max_num_patches`, rows (height, width, patch_size,
    # max_num_patches, target height, target width)
    table = NAFLEX_REF.ref["image_sizes"]
    @test all(c -> image_size_for_max_num_patches(c[1:4]...) == (c[5], c[6]), eachcol(table))
end

@testset "documented outputs" begin
    for (r, T, IP) in ((SIGLIP_REF, SiglipProcessor, SiglipImageProcessor),
        (NAFLEX_REF, Siglip2Processor, Siglip2ImageProcessor))
        cases = filter(!startswith("gemma/"), r.cases)
        outputs = map(name -> (((proc, inputs) = case(r, name)); proc(; inputs...)), cases)
        check_documented_keys(T, outputs)
        proc, inputs = case(r, "batch")
        check_documented_keys(IP, [proc.image_processor(inputs.images)])
    end
end

@testset "texts" begin
    proc = with_tokenizer(create_processor(SIGLIP), SIGLIP_TOKENIZER)
    ids = SIGLIP_TOKENIZER.ids["a cat"]
    # padded to 64 tokens with the padding token, and no attention mask
    out = proc(; text = "a cat")
    @test out == (; input_ids = reshape([ids; fill(1, 64 - length(ids))] .+ 1, 64, 1))
    @test size(proc(; text = ["a cat", "a red car"], max_length = 8).input_ids) == (8, 2)
    @test_throws "more than the 2 to pad to" proc(; text = "a cat", max_length = 2)
    # SigLIP 2 lowercases the texts, and truncates them to `max_length` with their last token
    naflex = with_tokenizer(create_processor(NAFLEX), NAFLEX_TOKENIZER)
    @test naflex(; text = "TWO DOGS, ON A SOFA!") == naflex(; text = "two dogs, on a sofa!")
    long = NAFLEX_TOKENIZER.ids[lowercase(case_texts(JSON.parse(NAFLEX_REF.meta["long"]))[1])]
    @test length(long) > 64
    @test naflex(; text = "a photo of " * join(fill("a small red cat", 20), " and "),
        max_length = 10).input_ids == reshape([long[1:9]; long[end]] .+ 1, 10, 1)
    @test_throws "`max_length` must be positive" naflex(; text = "a cat", max_length = 0)
    # images and text need their component
    for T in (SiglipProcessor, Siglip2Processor)
        p = T == SiglipProcessor ? proc : naflex
        @test_throws ["$(nameof(T)) has no tokenizer to encode text",
            "`using HuggingFaceTokenizers` before `create_processor`"] T(p.image_processor,
            nothing)(; text = "a cat")
        @test_throws "$(nameof(T)) has no image processor for `images`" T(nothing,
            p.tokenizer)(; images = rand(UInt8, 8, 8, 3))
        @test_throws "$(nameof(T)) needs `images` or `text`" p()
        @test_throws "`text` is a string or a vector of strings" p(; text = ids)
    end
    image = rand(UInt8, 12, 8, 3)
    @test SiglipProcessor(proc.image_processor, nothing)(; images = image) ==
          proc.image_processor(image)
end

@testset "settings" begin
    # the forms that transformers takes
    ip = SiglipImageProcessor(; size = (; height = 12, width = 16), resample = :bilinear,
        image_mean = 0.5, image_std = [0.5, 0.5, 0.5], rescale_factor = 1 // 255)
    @test ip.resample === 2 && ip.image_mean === 0.5 && ip.image_std === (0.5, 0.5, 0.5)
    @test size(ip(rand(UInt8, 13, 9, 3)).pixel_values) == (16, 12, 3, 1)
    nip = Siglip2ImageProcessor(; patch_size = 4.0, max_num_patches = 12, resample = "bicubic")
    @test nip.patch_size === 4 && nip.resample === 3
    for (T, kwargs, message) in (
        (SiglipImageProcessor, (; size = 12), "an integer `size` resizes the shortest edge"),
        (SiglipImageProcessor, (; do_center_crop = true), "`do_center_crop = true` is not supported"),
        (SiglipImageProcessor, (; resample = :lanczos), "`resample = 1` (lanczos) is not supported"),
        (Siglip2ImageProcessor, (; patch_size = 0), "`patch_size` must be positive"),
        (Siglip2ImageProcessor, (; max_num_patches = 2.5), "`max_num_patches` of Siglip2ImageProcessor is an integer"),
        (Siglip2ImageProcessor, (; image_std = (0.5, 0.5), do_convert_rgb = true), "`image_std` has one value per RGB channel"))
        @test_throws message T(; kwargs...)
    end
    # overrides by name, merged into nested settings
    proc = @test_logs create_processor(SIGLIP; size = (; width = 32))
    @test proc.image_processor.size == (; height = 224, width = 32)
    @test_throws "an integer `size`" create_processor(SIGLIP; size = 12)
    naflex = @test_logs create_processor(NAFLEX; max_num_patches = 64)
    @test naflex.image_processor.max_num_patches == 64
    @test_throws "`size` is not used by Siglip2ImageProcessor" create_processor(NAFLEX;
        size = (; height = 1, width = 1))
    @test_throws "no component of Siglip2Processor has a setting `max_length`" create_processor(
        NAFLEX; max_length = 32)
    text = sprint(show, MIME("text/plain"), naflex)
    @test occursin(r"max_num_patches = 64 +override \(preprocessor_config\.json: 256\)", text)
    @test occursin(r"do_convert_rgb = false +class default", text)
    @test endswith(text, "\n  tokenizer: nothing")
    # images without resizing, whose sides are multiples of the patches, in at most
    # `max_num_patches`, of one number of channels
    p = Siglip2ImageProcessor(; patch_size = 4, max_num_patches = 6, do_resize = false)
    @test p(rand(UInt8, 12, 8, 3)).spatial_shapes == [2; 3;;]
    @test_throws "multiples of `patch_size` (4), got 10 × 8" p(rand(UInt8, 10, 8, 3))
    @test_throws "more than `max_num_patches = 6`" p(rand(UInt8, 16, 8, 3))
    @test_throws "different numbers of channels" Siglip2ImageProcessor(; patch_size = 4,
        max_num_patches = 6, image_mean = 0.5, image_std = 0.5)([rand(UInt8, 8, 8, 3),
        rand(UInt8, 8, 8)])
    @test_throws "channels" Siglip2ImageProcessor(; patch_size = 4)(rand(UInt8, 8, 8))
end

# The tokenizers of the repositories, HuggingFaceTokenizers' of their tokenizer.json, which
# METALHEAD_TEST_NETWORK adds to the test environment (runtests.jl)
const TEST_NETWORK = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))
if TEST_NETWORK
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers, Tokenizer
    @testset "HuggingFaceTokenizers" begin
        for r in (SIGLIP_REF, NAFLEX_REF), name in r.cases
            proc, inputs = case(r, name; network = true)
            @testset "$name" begin
                @test proc.tokenizer isa Tokenizer
                check_outputs(r, name, proc(; inputs...))
            end
        end
        # the stand-ins give the ids of the tokenizers, but for texts longer than the
        # truncation of the NaFlex tokenizer.json, 64 tokens
        for (repo, standin) in ("google/siglip-base-patch16-224" => SIGLIP_TOKENIZER,
            "google/siglip2-base-patch16-224" => GEMMA_TOKENIZER,
            "google/siglip2-base-patch16-naflex" => NAFLEX_TOKENIZER)
            tok = create_processor(repo).tokenizer
            @test Metalhead.pad_token_id(tok) == standin.pad
            for (text, ids) in standin.ids
                length(ids) <= 64 && @test Metalhead.encode_ids(tok, text) == ids
            end
        end
        # SigLIP's tokenizer.json keeps `/`, `<` and `>`, which transformers'
        # SiglipTokenizer drops
        tok = create_processor("google/siglip-base-patch16-224").tokenizer
        @test Metalhead.encode_ids(tok, "a cat/dog") != Metalhead.encode_ids(tok, "a catdog")
    end
end
