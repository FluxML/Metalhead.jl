# The machinery of processors (src/hub/processors.jl) on classes of its own, which no
# repository names: an image processor, a tokenizer without a backend, and a processor
# that combines them, loaded from configuration files in temporary directories.
using Metalhead, Test
using JSON: JSON
using Metalhead: AbstractProcessor, AbstractImageProcessor, @processor, ProcessorFiles,
                 processor_from_entries, json_entries, read_json, register_processor!,
                 check_inputs
import Metalhead: processor_components, ignored_keys, restricted_keys, legacy_keys,
                  processor_fields, check_processor, load_processor, instantiate_processor,
                  architecture_of

include(joinpath(@__DIR__, "..", "processors_setup.jl"))

"""
    MetalheadTestImageProcessor(; do_resize = true, size = (; height = 4, width = 4), ...)

An image processor of the tests. Called on one image or a vector of them, it returns:

| Key            | Size               | Contents                                           |
|:-------------- |:------------------ |:-------------------------------------------------- |
| `pixel_values` | `(W, H, 3, batch)` | the images, rescaled, `Float32`                    |
| `pixel_mask`   | `(W, H, batch)`    | 1 for the pixels of the images; only with `do_pad` |
"""
@processor struct MetalheadTestImageProcessor <: AbstractImageProcessor
    do_resize::Bool = true
    size::NamedTuple = (; height = 4, width = 4)
    filter::String = "bilinear"
    rescale_factor::Float64 = 1 / 255
    image_mean::NTuple{3, Float64} = (0.5, 0.5, 0.5)
    do_pad::Bool = false
    provenance::NamedTuple = (;)
end

function processor_fields(::Type{MetalheadTestImageProcessor}, args::Tuple)
    do_resize, size, filter, rescale_factor, image_mean, do_pad, provenance = args
    size isa Integer && (size = (; height = size, width = size))
    size isa NamedTuple ||
        throw(ArgumentError("`size` is an integer or a NamedTuple, got $(repr(size))"))
    image_mean isa Real && (image_mean = ntuple(Returns(image_mean), 3))
    filter isa Symbol && (filter = String(filter))
    return (do_resize, size, filter, rescale_factor, image_mean, do_pad, provenance)
end
function check_processor(ip::MetalheadTestImageProcessor)
    ip.filter in ("bilinear", "bicubic") ||
        error("`filter = $(repr(ip.filter))` is not supported by " *
              "MetalheadTestImageProcessor (supported: \"bilinear\", \"bicubic\")")
    return nothing
end
function ignored_keys(::Type{MetalheadTestImageProcessor})
    return (; crop_size = "only used with `do_center_crop`")
end
restricted_keys(::Type{MetalheadTestImageProcessor}) = (; do_center_crop = (false,))
function legacy_keys(::Type{MetalheadTestImageProcessor})
    return (; max_height = (:size, :height), scale = (:rescale_factor,))
end

function (ip::MetalheadTestImageProcessor)(images)
    images = images isa AbstractVector ? images : [images]
    pixel_values = stack([Float32.(x .* ip.rescale_factor) for x in images])
    ip.do_pad || return (; pixel_values)
    pixel_mask = ones(Int, size(pixel_values, 1), size(pixel_values, 2), length(images))
    return (; pixel_values, pixel_mask)
end

"""
    MetalheadTestTokenizer(; model_max_length = 8, padding_side = "right", ...)

A tokenizer of the tests, of token ids only. Called on a vector of them, it returns:

| Key              | Size               | Contents                                   |
|:---------------- |:------------------ |:------------------------------------------ |
| `input_ids`      | `(seq_len, batch)` | the ids plus 1, padded with `pad_token_id` |
| `attention_mask` | `(seq_len, batch)` | 1 for the tokens, 0 for the padding        |
"""
@processor struct MetalheadTestTokenizer <: AbstractProcessor
    model_max_length::Int = 8
    padding_side::String = "right"
    pad_token::Union{Nothing, String} = "<pad>"
    pad_token_id::Int = 0
    chat_template::Union{Nothing, String} = nothing
    added_tokens::Dict{String, Any} = Dict{String, Any}()
    provenance::NamedTuple = (;)
end

function processor_fields(::Type{MetalheadTestTokenizer}, args::Tuple)
    added_tokens = args[6]
    added_tokens isa NamedTuple &&
        (added_tokens = Dict{String, Any}(String(k) => v for (k, v) in pairs(added_tokens)))
    return (args[1:5]..., added_tokens, args[7])
end
function ignored_keys(::Type{MetalheadTestTokenizer})
    return (; clean_up_tokenization_spaces = "only changes decoding")
end

function load_processor(::Type{MetalheadTestTokenizer}, files::ProcessorFiles; kwargs...)
    dict = read_json(files, "tokenizer_config.json")
    dict === nothing && error("no tokenizer_config.json")
    entries = json_entries(dict, "tokenizer_config.json")
    return processor_from_entries(MetalheadTestTokenizer, entries; kwargs...)
end

function (tok::MetalheadTestTokenizer)(ids::AbstractVector{<:AbstractVector{<:Integer}})
    n = maximum(length, ids)
    input_ids = fill(tok.pad_token_id + 1, n, length(ids))
    attention_mask = zeros(Int, n, length(ids))
    for (b, x) in enumerate(ids)
        r = tok.padding_side == "right" ? (1:length(x)) : ((n - length(x) + 1):n)
        input_ids[r, b] = x .+ 1
        attention_mask[r, b] .= 1
    end
    return (; input_ids, attention_mask)
end

"""
    MetalheadTestProcessor(image_processor, tokenizer; padding_side = "right",
                           num_image_tokens = 4)

A processor of the tests that combines a `MetalheadTestImageProcessor` and a
`MetalheadTestTokenizer`. Called as `proc(; images, text)`, it returns:

| Key              | Size               | Contents                                 |
|:---------------- |:------------------ |:---------------------------------------- |
| `pixel_values`   | `(W, H, 3, batch)` | the images; with `images`                |
| `pixel_mask`     | `(W, H, batch)`    | only with `images` and `do_pad`          |
| `input_ids`      | `(seq_len, batch)` | the token indices of `text`; with `text` |
| `attention_mask` | `(seq_len, batch)` | 0 for the padding; with `text`           |
"""
@processor (image_processor, tokenizer) struct MetalheadTestProcessor <: AbstractProcessor
    image_processor::Any = nothing
    tokenizer::Any = nothing
    padding_side::String = "right"
    num_image_tokens::Int = 4
    provenance::NamedTuple = (;)
end

function processor_components(::Type{MetalheadTestProcessor})
    return (; image_processor = MetalheadTestImageProcessor,
        tokenizer = MetalheadTestTokenizer)
end
ignored_keys(::Type{MetalheadTestProcessor}) = (; video_processor = "videos aren't ported")

function (p::MetalheadTestProcessor)(; images = nothing, text = nothing)
    out = (;)
    images === nothing || (out = merge(out, p.image_processor(images)))
    text === nothing || (out = merge(out, p.tokenizer(text)))
    return out
end

# The names these tests register, removed at the end, so that the other test files of the
# worker find the registries as they were
const CLASSES = ["MetalheadTestImageProcessor", "MetalheadTestTokenizer",
    "MetalheadTestProcessor", "MetalheadTestVideoProcessor", "MetalheadTestOther"]
const MODEL_TYPES = ["metalhead_test"]

register_processor!("MetalheadTestImageProcessor", MetalheadTestImageProcessor)
register_processor!("MetalheadTestTokenizer", MetalheadTestTokenizer)
register_processor!("MetalheadTestProcessor", MetalheadTestProcessor;
    aliases = ["MetalheadTestVideoProcessor"], model_types = MODEL_TYPES)

# A directory with the JSON files `files` (name => Dict), in subfolders for names with "/"
function with_files(f, files)
    mktempdir() do dir
        for (name, content) in files
            mkpath(dirname(joinpath(dir, name)))
            write(joinpath(dir, name), JSON.json(content))
        end
        return f(dir)
    end
end

const IMAGE = Dict("image_processor_type" => "MetalheadTestImageProcessor",
    "size" => Dict("height" => 6, "width" => 6), "filter" => "bicubic")
const TOKENIZER = Dict("model_max_length" => 16, "padding_side" => "left")
const PROCESSOR = Dict("processor_class" => "MetalheadTestProcessor",
    "num_image_tokens" => 6)
const COMPOSITE = ["processor_config.json" => PROCESSOR,
    "preprocessor_config.json" => IMAGE, "tokenizer_config.json" => TOKENIZER]

with_class(dict, class) = merge(dict, Dict("processor_class" => class))
plain(p) = sprint(show, MIME("text/plain"), p)
files_of(dir) = ProcessorFiles(Metalhead.model_source(dir, nothing))

# The class that AutoProcessor's lookup finds for a tokenizer without `processor_class`:
# transformers' TokenizersBackend, which may not be ported
function fallback_class(dir)
    try
        return nameof(Metalhead.auto_processor_class(files_of(dir), nothing))
    catch e
        unported = "`TokenizersBackend` (a tokenizer) has not been ported"
        e isa ErrorException && occursin(unported, e.msg) || rethrow()
        return :TokenizersBackend
    end
end

struct MetalheadTestModel end

# A library whose config.json carries the preprocessing, and one whose config.json doesn't
struct MetalheadTestFormat <: Metalhead.AbstractConfigFormat end
struct MetalheadTestPlainFormat <: Metalhead.AbstractConfigFormat end
function architecture_of(::MetalheadTestFormat, config)
    return get(config, "metalhead_test_architecture", nothing)
end
function architecture_of(::MetalheadTestPlainFormat, config)
    return get(config, "metalhead_test_plain_architecture", nothing)
end
function instantiate_processor(::MetalheadTestFormat, files, config; kwargs...)
    entries = json_entries(config["preprocessing"], "config.json")
    return processor_from_entries(MetalheadTestImageProcessor, entries; kwargs...)
end

try
    @testset "classes" begin
        ip = MetalheadTestImageProcessor()
        @test ip.size == (; height = 4, width = 4) && ip.provenance == (;)
        # the constructors take the forms of `processor_fields` and run `check_processor`
        ip = MetalheadTestImageProcessor(; size = 8, image_mean = 0, filter = :bicubic)
        @test ip.size == (; height = 8, width = 8)
        @test ip.image_mean === (0.0, 0.0, 0.0)
        @test ip.filter == "bicubic"
        @test_throws ArgumentError MetalheadTestImageProcessor(; size = "large")
        message = "`filter = \"nearest\"` is not supported by " *
                  "MetalheadTestImageProcessor (supported: \"bilinear\", \"bicubic\")"
        @test_throws message MetalheadTestImageProcessor(; filter = "nearest")
        @test_throws "`filter = \"box\"` is not supported" MetalheadTestImageProcessor(true,
            (; height = 2, width = 2), "box", 1.0, (0.0, 0.0, 0.0), false, (;))
        tok = MetalheadTestTokenizer(; added_tokens = (; a = 1))
        @test tok.added_tokens == Dict("a" => 1)

        # the components of a processor, and its constructor from them
        @test processor_components(MetalheadTestImageProcessor) == (;)
        p = MetalheadTestProcessor(ip, tok; num_image_tokens = 2)
        @test p.image_processor === ip && p.tokenizer === tok && p.num_image_tokens == 2
        @test MetalheadTestProcessor().image_processor === nothing
        # the type of the component is printed with its module, this test's
        @test_throws ["the `image_processor` of MetalheadTestProcessor is a " *
                      "MetalheadTestImageProcessor (or `nothing`), got a ",
            "MetalheadTestTokenizer"] MetalheadTestProcessor(tok, tok)
        @test Metalhead.settings(p) == (; padding_side = "right", num_image_tokens = 2)
        @test Metalhead.components(p) == (; image_processor = ip, tokenizer = tok)
    end

    @testset "registry" begin
        type(name) = Metalhead.processor_type(name, "")
        @test type("MetalheadTestProcessor") === MetalheadTestProcessor
        # aliases, and the suffixes of transformers' fast and PIL image processors
        @test type("MetalheadTestVideoProcessor") === MetalheadTestProcessor
        @test type("MetalheadTestImageProcessorFast") === MetalheadTestImageProcessor
        @test type("MetalheadTestImageProcessorPil") === MetalheadTestImageProcessor
        @test_throws ["the processor class `MetalheadTestUnported` (named here)",
            "has not been ported to Metalhead (ported: "] Metalhead.processor_type(
            "MetalheadTestUnported", "named here")
        # registering a class again is a no-op; another class under its name is an error
        @test register_processor!("MetalheadTestProcessor", MetalheadTestProcessor) ===
              MetalheadTestProcessor
        message = "processor class `MetalheadTestProcessor` is already registered as"
        @test_throws message register_processor!("MetalheadTestProcessor",
            MetalheadTestTokenizer)
        message = "processor class `MetalheadTestVideoProcessor` is already registered"
        @test_throws message register_processor!("MetalheadTestOther",
            MetalheadTestTokenizer; aliases = ["MetalheadTestVideoProcessor"])
    end

    @testset "AutoProcessor's lookup" begin
        # `processor_class` in processor_config.json, whose `image_processor` wins over
        # preprocessor_config.json
        processor = merge(PROCESSOR,
            Dict("image_processor" => merge(IMAGE, Dict("filter" => "bilinear"))))
        files = ["processor_config.json" => processor, "preprocessor_config.json" => IMAGE,
            "tokenizer_config.json" => TOKENIZER]
        with_files(files) do dir
            p = @test_logs create_processor(dir)
            @test p isa MetalheadTestProcessor
            @test p.image_processor.filter == "bilinear"
            @test p.tokenizer.padding_side == "left"
            @test occursin(r"filter = \"bilinear\" +processor_config\.json", plain(p))
        end
        # in preprocessor_config.json, whose image processor it holds, over the stale class
        # of tokenizer_config.json
        image = with_class(IMAGE, "MetalheadTestProcessor")
        unported = with_class(TOKENIZER, "MetalheadTestUnported")
        with_files(["preprocessor_config.json" => image,
            "tokenizer_config.json" => unported]) do dir
            p = create_processor(dir)
            @test p isa MetalheadTestProcessor
            @test occursin(r"filter = \"bicubic\" +preprocessor_config\.json", plain(p))
        end
        # in video_preprocessor_config.json, without a preprocessor_config.json
        video = Dict("processor_class" => "MetalheadTestVideoProcessor")
        with_files(["processor_config.json" => Dict("image_processor" => IMAGE),
            "video_preprocessor_config.json" => video,
            "tokenizer_config.json" => TOKENIZER]) do dir
            @test create_processor(dir) isa MetalheadTestProcessor
        end
        # in tokenizer_config.json, and in config.json
        tokenizer = with_class(TOKENIZER, "MetalheadTestTokenizer")
        with_files(["tokenizer_config.json" => tokenizer]) do dir
            @test create_processor(dir) isa MetalheadTestTokenizer
        end
        config = Dict("processor_class" => "MetalheadTestProcessor")
        with_files(["config.json" => config, "preprocessor_config.json" => IMAGE,
            "tokenizer_config.json" => TOKENIZER]) do dir
            @test create_processor(dir) isa MetalheadTestProcessor
        end
        # by the model type of config.json
        config = Dict("architectures" => ["MetalheadTestModel"],
            "model_type" => "metalhead_test")
        with_files(["config.json" => config, "preprocessor_config.json" => IMAGE,
            "tokenizer_config.json" => TOKENIZER]) do dir
            @test create_processor(dir) isa MetalheadTestProcessor
        end
        # else a tokenizer, before an image processor
        with_files(["preprocessor_config.json" => IMAGE,
            "tokenizer_config.json" => TOKENIZER]) do dir
            @test fallback_class(dir) === :TokenizersBackend
        end
        # else the image processor of `image_processor_type`
        fast = merge(IMAGE,
            Dict("image_processor_type" => "MetalheadTestImageProcessorFast"))
        with_files(["preprocessor_config.json" => fast]) do dir
            @test create_processor(dir) isa MetalheadTestImageProcessor
        end
        # in a subfolder
        with_files(["tokenizer/tokenizer_config.json" => tokenizer]) do dir
            @test create_processor(dir; subfolder = "tokenizer").model_max_length == 16
            message = "has no processor files (processor_config.json, " *
                      "preprocessor_config.json, tokenizer_config.json or tokenizer.json)"
            @test_throws message create_processor(dir)
        end

        with_files(["config.json" => Dict("model_type" => "metalhead_test"),
            "tokenizer_config.json" => TOKENIZER]) do dir
            message = "has no image processor (preprocessor_config.json, or " *
                      "`image_processor` in processor_config.json)"
            @test_throws message create_processor(dir)
        end
        with_files(["tokenizer_config.json" => unported]) do dir
            message = "the processor class `MetalheadTestUnported` (`processor_class` in " *
                      "tokenizer_config.json) has not been ported to Metalhead"
            @test_throws message create_processor(dir)
        end
        image = merge(IMAGE, Dict("image_processor_type" => "MetalheadTestTokenizer"))
        files = ["processor_config.json" => PROCESSOR, "preprocessor_config.json" => image,
            "tokenizer_config.json" => TOKENIZER]
        with_files(files) do dir
            message = "the image processor of preprocessor_config.json is a " *
                      "MetalheadTestTokenizer, not a MetalheadTestImageProcessor"
            @test_throws message create_processor(dir)
        end
    end

    @testset "strict settings" begin
        # an unknown key warns once for its class; ignored and service keys are skipped
        config = merge(IMAGE, Dict("metalhead_test_unknown" => 1, "crop_size" => 3,
            "auto_map" => Dict("AutoImageProcessor" => "remote.Class")))
        with_files(["preprocessor_config.json" => config]) do dir
            message = "`metalhead_test_unknown` in preprocessor_config.json is not a " *
                      "setting of MetalheadTestImageProcessor; it is ignored"
            @test_logs (:warn, message) create_processor(dir)
            @test_logs create_processor(dir)
        end
        # a restricted key with an allowed value, in a file or an override
        unsupported = "`do_center_crop = true` is not supported by " *
                      "MetalheadTestImageProcessor"
        config = merge(IMAGE, Dict("do_center_crop" => false))
        with_files(["preprocessor_config.json" => config]) do dir
            @test (@test_logs create_processor(dir; do_center_crop = false)) isa
                  MetalheadTestImageProcessor
            @test_throws [unsupported, "(supported: false)"] create_processor(dir;
                do_center_crop = true)
        end
        config = merge(IMAGE, Dict("do_center_crop" => true))
        with_files(["preprocessor_config.json" => config]) do dir
            message = "`do_center_crop = true` in preprocessor_config.json is not " *
                      "supported by MetalheadTestImageProcessor (supported: false)"
            @test_throws message create_processor(dir)
        end
        # an override of an ignored key gives the reason, of another key the closest setting
        with_files(["preprocessor_config.json" => IMAGE]) do dir
            message = "`crop_size` is not used by MetalheadTestImageProcessor: " *
                      "only used with `do_center_crop`"
            @test_throws message create_processor(dir; crop_size = 3)
            message = "MetalheadTestImageProcessor has no setting `do_resze` " *
                      "(MetalheadTestImageProcessor has `do_resize`)"
            @test_throws message create_processor(dir; do_resze = false)
            message = "MetalheadTestImageProcessor has no setting `colour`"
            @test_throws message create_processor(dir; colour = 1)
        end
        # a legacy key sets the entry at its path, and drops its own entry from the setting
        size = Dict("height" => 6, "width" => 6, "max_height" => 7)
        config = merge(IMAGE, Dict("max_height" => 9, "scale" => 0.5, "size" => size))
        with_files(["preprocessor_config.json" => config]) do dir
            ip = create_processor(dir)
            @test ip.size == (; height = 9, width = 6)
            @test ip.rescale_factor == 0.5
            text = plain(ip)
            @test occursin(r"size = \(height = 9, width = 6\) +preprocessor_config\.json " *
                           r"\(`max_height`\)", text)
            @test occursin(r"rescale_factor = 0\.5 +preprocessor_config\.json \(`scale`\)",
                text)
            ip = create_processor(dir; max_height = 3)
            @test ip.size == (; height = 3, width = 6)
            @test occursin("size = (height = 3, width = 6)  override " *
                           "(preprocessor_config.json (`max_height`): " *
                           "(height = 9, width = 6))", plain(ip))
        end
        config = merge(IMAGE, Dict("size" => size))
        with_files(["preprocessor_config.json" => config]) do dir
            @test create_processor(dir; max_height = 3).size == (; height = 3, width = 6)
        end
    end

    @testset "null and later entries" begin
        # `null` in a file is missing; `Some(nothing)` sets `nothing`
        config = merge(with_class(TOKENIZER, "MetalheadTestTokenizer"),
            Dict("pad_token" => nothing, "chat_template" => nothing))
        with_files(["tokenizer_config.json" => config]) do dir
            tok = create_processor(dir)
            @test tok.pad_token == "<pad>" && tok.chat_template === nothing
            @test occursin(r"pad_token = \"<pad>\" +class default", plain(tok))
        end
        entries = [("pad_token", Some(nothing), "a.json"),
            ("padding_side", "left", "a.json"), ("padding_side", "right", "b.json"),
            ("model_max_length", nothing, "b.json")]
        tok = processor_from_entries(MetalheadTestTokenizer, entries)
        @test tok.pad_token === nothing && tok.padding_side == "right"
        @test tok.provenance.pad_token.source == "a.json"
        @test tok.provenance.padding_side.source == "b.json"
        @test tok.provenance.model_max_length.source == "class default"
        # the provenance holds the values as the constructor stored them
        ip = processor_from_entries(MetalheadTestImageProcessor,
            [("size", 5, "c.json"), ("filter", :bicubic, "c.json")])
        @test ip.provenance.size.value == (; height = 5, width = 5)
        @test ip.provenance.filter.value == "bicubic"
        @test_throws "MetalheadTestTokenizer has no setting `pad`" processor_from_entries(
            MetalheadTestTokenizer, entries; pad = 1)
    end

    @testset "overrides" begin
        with_files(COMPOSITE) do dir
            # each to the component that has a setting of its name, NamedTuples merged into
            # nested settings
            p = create_processor(dir; size = (; height = 8), model_max_length = 4,
                num_image_tokens = 2, pad_token = nothing)
            @test p.image_processor.size == (; height = 8, width = 6)
            @test p.tokenizer.model_max_length == 4 && p.tokenizer.pad_token === nothing
            @test p.num_image_tokens == 2
            p = create_processor(dir; max_height = 5, do_center_crop = false)
            @test p.image_processor.size == (; height = 5, width = 6)
            @test_throws ["`padding_side` is a setting of several components of " *
                          "MetalheadTestProcessor (tokenizer, self)",
                "set it on one of them after loading"] create_processor(dir;
                padding_side = "left")
            @test_throws ["no component of MetalheadTestProcessor has a setting " *
                          "`model_max_lenght` (MetalheadTestTokenizer has " *
                          "`model_max_length`)"] create_processor(dir; model_max_lenght = 4)
            message = "no component of MetalheadTestProcessor has a setting `colour`"
            @test_throws message create_processor(dir; colour = 1)
            @test_throws ["`video_processor` is not used by MetalheadTestProcessor: " *
                          "videos aren't ported"] create_processor(dir; video_processor = 1)
            @test_throws ["`crop_size` is not used by MetalheadTestImageProcessor: " *
                          "only used with `do_center_crop`"] create_processor(dir;
                crop_size = 3)
            @test_throws ["`do_center_crop = true` is not supported by " *
                          "MetalheadTestImageProcessor"] create_processor(dir;
                do_center_crop = true)
            @test_throws ArgumentError create_processor(dir; size = "large")
            @test_throws "`filter = \"nearest\"` is not supported" create_processor(dir;
                filter = "nearest")
        end
        routed = Metalhead.route_overrides(MetalheadTestProcessor,
            (; filter = "bilinear", pad_token = nothing, num_image_tokens = 1))
        @test routed.self == (; num_image_tokens = 1)
        @test routed.image_processor == (; filter = "bilinear")
        @test routed.tokenizer == (; pad_token = nothing)
    end

    @testset "provenance and show" begin
        with_files(COMPOSITE) do dir
            p = create_processor(dir; size = (; width = 2), filter = :bilinear)
            text = plain(p)
            @test startswith(text, "MetalheadTestProcessor\n  padding_side = ")
            @test occursin(r"\n  padding_side = \"right\" +class default\n", text)
            @test occursin(r"\n  num_image_tokens = 6 +processor_config\.json\n", text)
            @test occursin(r"\n  image_processor: MetalheadTestImageProcessor\n" *
                           r"    do_resize = true +class default\n", text)
            @test occursin("size = (height = 6, width = 2)  override " *
                           "(preprocessor_config.json: (height = 6, width = 6))\n", text)
            @test occursin(r"filter = \"bilinear\" +override " *
                           r"\(preprocessor_config\.json: \"bicubic\"\)\n", text)
            @test occursin(r"\n    do_pad = false +class default\n", text)
            @test occursin(r"\n  tokenizer: MetalheadTestTokenizer\n" *
                           r"    model_max_length = 16 +tokenizer_config\.json\n", text)
            # an override of a class default
            ip = create_processor(dir; do_pad = true).image_processor
            @test occursin(r"do_pad = true +override \(class default: false\)", plain(ip))
            # a processor changed after loading, or made by hand
            ip = MetalheadTestImageProcessor(; Metalhead.settings(ip)..., do_resize = false,
                provenance = ip.provenance)
            @test occursin(r"do_resize = false +set after loading", plain(ip))
            @test occursin(r"do_pad = true +override", plain(ip))
            @test rstrip.(split(plain(MetalheadTestImageProcessor()), '\n')) ==
                  ["MetalheadTestImageProcessor", "  do_resize = true",
                "  size = (height = 4, width = 4)", "  filter = \"bilinear\"",
                "  rescale_factor = 0.00392157", "  image_mean = (0.5, 0.5, 0.5)",
                "  do_pad = false"]
            @test sprint(show, p.tokenizer) ==
                  "MetalheadTestTokenizer(model_max_length = 16, " *
                  "padding_side = \"left\", pad_token = \"<pad>\", pad_token_id = 0, " *
                  "chat_template = nothing, added_tokens = 0 entries)"
            compact = sprint(show, p)
            @test startswith(compact, "MetalheadTestProcessor(image_processor = " *
                                      "MetalheadTestImageProcessor(do_resize = true, ")
            @test endswith(compact, "; padding_side = \"right\", num_image_tokens = 6)")
        end
        # long values are cut
        tok = MetalheadTestTokenizer(; chat_template = "x"^100)
        @test occursin("chat_template = \"" * "x"^70 * "…, added_tokens",
            sprint(show, tok))
    end

    @testset "files" begin
        with_files(["tokenizer_config.json" => TOKENIZER]) do dir
            files = files_of(dir)
            @test Metalhead.has_file(files, "tokenizer_config.json")
            @test !Metalhead.has_file(files, "processor_config.json")
            config = read_json(files, "tokenizer_config.json")
            @test config["padding_side"] == "left"
            @test read_json(files, "tokenizer_config.json") === config  # read once
            @test read_json(files, "processor_config.json") === nothing
            @test !Metalhead.has_file(files, "processor_config.json")
            touch(joinpath(dir, "vocab.txt"))
            f = Metalhead.SourceFile(files.source, "vocab.txt")
            @test Metalhead.local_path(f) == joinpath(dir, "vocab.txt")
            @test Metalhead.local_path_if_exists(f) == joinpath(dir, "vocab.txt")
            @test endswith(sprint(show, f), "vocab.txt")
            @test Metalhead.local_path("path/to/vocab.txt") == "path/to/vocab.txt"
            absent = Metalhead.SourceFile(files.source, "merges.txt")
            @test_throws "has no merges.txt" Metalhead.local_path(absent)
            @test Metalhead.local_path_if_exists(absent) === nothing
        end
    end

    @testset "the processor of a config.json" begin
        specs = [
            Metalhead.ArchSpec(MetalheadTestFormat(), "MetalheadTestArchitecture",
                () -> nothing),
            Metalhead.ArchSpec(MetalheadTestPlainFormat(), "MetalheadTestPlainArchitecture",
                () -> nothing)]
        foreach(Metalhead.register!, specs)
        try
            preprocessing = Dict("filter" => "bicubic", "metalhead_test_cfg" => 1)
            config = Dict("metalhead_test_architecture" => "MetalheadTestArchitecture",
                "preprocessing" => preprocessing)
            with_files(["config.json" => config]) do dir
                unknown = (:warn, r"`metalhead_test_cfg` in config\.json")
                ip = @test_logs unknown create_processor(dir; do_pad = true)
                @test ip isa MetalheadTestImageProcessor
                @test ip.filter == "bicubic" && ip.do_pad
                @test occursin(r"filter = \"bicubic\" +config\.json", plain(ip))
                @test occursin(r"do_pad = true +override \(class default: false\)",
                    plain(ip))
                @test instantiate_processor(MetalheadTestPlainFormat(), files_of(dir),
                    config) === nothing
            end
            # without a method, AutoProcessor's lookup
            config = Dict(
                "metalhead_test_plain_architecture" => "MetalheadTestPlainArchitecture")
            files = ["config.json" => config, "preprocessor_config.json" => IMAGE]
            with_files(files) do dir
                @test create_processor(dir) isa MetalheadTestImageProcessor
            end
        finally
            foreach(spec -> delete!(Metalhead.REGISTRY, spec.name), specs)
        end
    end

    @testset "inputs" begin
        m = MetalheadTestModel()
        @test check_inputs(m, (; a = 1, b = 2), (:a,)) === nothing
        @test_throws ArgumentError check_inputs(m, (; b = 2), (:a,))
        message = "MetalheadTestModel needs the inputs `a` and `c`, but `inputs` has no " *
                  "`c` (it has `a` and `b`); pass the output of its processor"
        @test_throws message check_inputs(m, (; a = 1, b = 2), (:a, :c);
            hint = "pass the output of its processor")
        @test_throws "has no `a` (it has no keys)" check_inputs(m, (;), [:a])
    end

    @testset "documented outputs" begin
        @test documented_keys(MetalheadTestImageProcessor) == [:pixel_values, :pixel_mask]
        with_files(COMPOSITE) do dir
            p = create_processor(dir)
            q = create_processor(dir; do_pad = true)
            image = rand(UInt8, 6, 6, 3)
            ids = [[1, 2, 3], [4]]
            out = p(; images = [image, image], text = ids)
            @test size(out.pixel_values) == (6, 6, 3, 2)
            @test out.input_ids == [2 1; 3 1; 4 5]  # padded on the left
            check_documented_keys(MetalheadTestImageProcessor,
                [p.image_processor(image), q.image_processor([image])])
            check_documented_keys(MetalheadTestTokenizer, [p.tokenizer(ids)])
            check_documented_keys(MetalheadTestProcessor,
                [out, q(; images = image), p(; text = ids)])
        end
    end
finally
    foreach(name -> delete!(Metalhead.PROCESSOR_CLASSES, name), CLASSES)
    foreach(type -> delete!(Metalhead.PROCESSOR_MODEL_TYPES, type), MODEL_TYPES)
end
