# The machinery of processors (src/hub/processors.jl) on classes of its own, which no
# repository names: an image processor, a video processor, and a processor that combines
# them, loaded from configuration files in temporary directories.
using Metalhead, Test
using JSON: JSON
using Metalhead: AbstractProcessor, AbstractImageProcessor, @processor, ProcessorFiles,
                 processor_from_entries, json_entries, read_json, register_processor!,
                 check_inputs
import Metalhead: ignored_keys, legacy_keys, standardize_settings, load_processor,
                  instantiate_processor, architecture_of

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
    do_center_crop::Bool = false
    do_pad::Bool = false
    provenance::NamedTuple = (;)
end

function standardize_settings(::Type{MetalheadTestImageProcessor}, s::NamedTuple)
    size = s.size isa Integer ? (; height = s.size, width = s.size) : s.size
    size isa NamedTuple ||
        throw(ArgumentError("`size` is an integer or a NamedTuple, got $(repr(size))"))
    image_mean = s.image_mean isa Real ? ntuple(Returns(s.image_mean), 3) : s.image_mean
    filter = s.filter isa Symbol ? String(s.filter) : s.filter
    filter in ("bilinear", "bicubic") ||
        throw(ArgumentError("`filter = $(repr(filter))` is not supported by " *
                            "MetalheadTestImageProcessor " *
                            "(supported: \"bilinear\", \"bicubic\")"))
    isequal(s.do_center_crop, false) ||
        throw(ArgumentError("`do_center_crop = $(repr(s.do_center_crop))` is not " *
                            "supported by MetalheadTestImageProcessor (supported: false)"))
    return merge(s, (; size, image_mean, filter))
end
function ignored_keys(::Type{MetalheadTestImageProcessor})
    return (; crop_size = "only used with `do_center_crop`")
end
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
    MetalheadTestVideoProcessor(; num_frames = 8, padding_side = "right", ...)

A video processor of the tests, of videos as vectors of frames. Called on a vector of them,
it returns:

| Key                   | Size              | Contents                             |
|:--------------------- |:----------------- |:------------------------------------ |
| `pixel_values_videos` | `(frames, batch)` | the frames, padded with 0, `Float32` |
| `video_mask`          | `(frames, batch)` | 1 for the frames, 0 for the padding  |
"""
@processor struct MetalheadTestVideoProcessor <: AbstractProcessor
    num_frames::Int = 8
    padding_side::String = "right"
    fps::Union{Nothing, Float64} = 2.0
    video_metadata::Dict{String, Any} = Dict{String, Any}()
    provenance::NamedTuple = (;)
end

function standardize_settings(::Type{MetalheadTestVideoProcessor}, s::NamedTuple)
    s.video_metadata isa NamedTuple || return s
    metadata = Dict{String, Any}(String(k) => v for (k, v) in pairs(s.video_metadata))
    return merge(s, (; video_metadata = metadata))
end

# A class that isn't an image processor loads with a method of its own
function load_processor(::Type{MetalheadTestVideoProcessor}, files::ProcessorFiles;
        kwargs...)
    dict = read_json(files, "video_preprocessor_config.json")
    dict === nothing && error("no video_preprocessor_config.json")
    entries = json_entries(dict, "video_preprocessor_config.json")
    return processor_from_entries(MetalheadTestVideoProcessor, entries; kwargs...)
end

function (vp::MetalheadTestVideoProcessor)(videos::AbstractVector{<:AbstractVector})
    n = maximum(length, videos)
    pixel_values_videos = zeros(Float32, n, length(videos))
    video_mask = zeros(Int, n, length(videos))
    for (b, x) in enumerate(videos)
        r = vp.padding_side == "right" ? (1:length(x)) : ((n - length(x) + 1):n)
        pixel_values_videos[r, b] = x
        video_mask[r, b] .= 1
    end
    return (; pixel_values_videos, video_mask)
end

"""
    MetalheadTestProcessor(image_processor, video_processor; padding_side = "right",
                           num_image_tokens = 4)

A processor of the tests that combines a `MetalheadTestImageProcessor` and a
`MetalheadTestVideoProcessor`. Called as `proc(; images, videos)`, it returns:

| Key                   | Size               | Contents                         |
|:--------------------- |:------------------ |:-------------------------------- |
| `pixel_values`        | `(W, H, 3, batch)` | the images; with `images`        |
| `pixel_mask`          | `(W, H, batch)`    | only with `images` and `do_pad`  |
| `pixel_values_videos` | `(frames, batch)`  | the videos; with `videos`        |
| `video_mask`          | `(frames, batch)`  | 0 for the padding; with `videos` |
"""
@processor (image_processor, video_processor) struct MetalheadTestProcessor <:
                                                     AbstractProcessor
    image_processor::Union{Nothing, MetalheadTestImageProcessor} = nothing
    video_processor::Union{Nothing, MetalheadTestVideoProcessor} = nothing
    padding_side::String = "right"
    num_image_tokens::Int = 4
    provenance::NamedTuple = (;)
end

ignored_keys(::Type{MetalheadTestProcessor}) = (; feature_extractor = "audio isn't ported")

function (p::MetalheadTestProcessor)(; images = nothing, videos = nothing)
    out = (;)
    images === nothing || (out = merge(out, p.image_processor(images)))
    videos === nothing || (out = merge(out, p.video_processor(videos)))
    return out
end

# The names these tests register, removed at the end, so that the other test files of the
# worker find the registries as they were
const CLASSES = ["MetalheadTestImageProcessor", "MetalheadTestProcessor",
    "MetalheadTestAliasProcessor", "MetalheadTestOther"]
const MODEL_TYPES = ["metalhead_test"]

register_processor!("MetalheadTestImageProcessor", MetalheadTestImageProcessor)
register_processor!("MetalheadTestProcessor", MetalheadTestProcessor;
    aliases = ["MetalheadTestAliasProcessor"], model_types = MODEL_TYPES)

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
const VIDEO = Dict("num_frames" => 16, "padding_side" => "left")
const PROCESSOR = Dict("processor_class" => "MetalheadTestProcessor",
    "num_image_tokens" => 6)
const COMPOSITE = ["processor_config.json" => PROCESSOR,
    "preprocessor_config.json" => IMAGE, "video_preprocessor_config.json" => VIDEO]

with_class(dict, class) = merge(dict, Dict("processor_class" => class))
plain(p) = sprint(show, MIME("text/plain"), p)
files_of(dir) = ProcessorFiles(Metalhead.model_source(dir, nothing))

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
        # the constructors take the forms of `standardize_settings`, and reject the values
        # that the port doesn't implement
        ip = MetalheadTestImageProcessor(; size = 8, image_mean = 0, filter = :bicubic)
        @test ip.size == (; height = 8, width = 8)
        @test ip.image_mean === (0.0, 0.0, 0.0)
        @test ip.filter == "bicubic"
        @test_throws ArgumentError MetalheadTestImageProcessor(; size = "large")
        message = "`filter = \"nearest\"` is not supported by " *
                  "MetalheadTestImageProcessor (supported: \"bilinear\", \"bicubic\")"
        @test_throws ArgumentError(message) MetalheadTestImageProcessor(;
            filter = "nearest")
        @test_throws "`filter = \"box\"` is not supported" MetalheadTestImageProcessor(true,
            (; height = 2, width = 2), "box", 1.0, (0.0, 0.0, 0.0), false, false, (;))
        message = "`do_center_crop = true` is not supported by " *
                  "MetalheadTestImageProcessor (supported: false)"
        @test_throws message MetalheadTestImageProcessor(; do_center_crop = true)
        vp = MetalheadTestVideoProcessor(; video_metadata = (; a = 1))
        @test vp.video_metadata == Dict("a" => 1)
        # by default, the settings as they are, converted to the types of the fields
        @test MetalheadTestProcessor(; num_image_tokens = 2.0).num_image_tokens === 2

        # the components of a processor, of the classes of their fields, and its
        # constructor from them
        @test Metalhead.component_names(MetalheadTestImageProcessor) == ()
        @test Metalhead.component_classes(MetalheadTestProcessor) ==
              (; image_processor = MetalheadTestImageProcessor,
            video_processor = MetalheadTestVideoProcessor)
        p = MetalheadTestProcessor(ip, vp; num_image_tokens = 2)
        @test p.image_processor === ip && p.video_processor === vp
        @test p.num_image_tokens == 2
        @test MetalheadTestProcessor().image_processor === nothing
        # the type of the component is printed with its module, this test's
        @test_throws ["the `image_processor` of MetalheadTestProcessor is a " *
                      "MetalheadTestImageProcessor (or `nothing`), got a ",
            "MetalheadTestVideoProcessor"] MetalheadTestProcessor(vp, vp)
        message = "the `video_processor` of MetalheadTestProcessor is a " *
                  "MetalheadTestVideoProcessor (or `nothing`), got a Int64"
        @test_throws message MetalheadTestProcessor(; video_processor = 1)
        @test Metalhead.settings(p) == (; padding_side = "right", num_image_tokens = 2)
        @test Metalhead.components(p) == (; image_processor = ip, video_processor = vp)
    end

    @testset "registry" begin
        type(name) = Metalhead.processor_type(name, "")
        @test type("MetalheadTestProcessor") === MetalheadTestProcessor
        # aliases, and the suffixes of transformers' fast and PIL image processors
        @test type("MetalheadTestAliasProcessor") === MetalheadTestProcessor
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
            MetalheadTestImageProcessor)
        message = "processor class `MetalheadTestAliasProcessor` is already registered"
        @test_throws message register_processor!("MetalheadTestOther",
            MetalheadTestImageProcessor; aliases = ["MetalheadTestAliasProcessor"])
    end

    @testset "AutoProcessor's lookup" begin
        # `processor_class` in processor_config.json, whose `image_processor` wins over
        # preprocessor_config.json
        processor = merge(PROCESSOR,
            Dict("image_processor" => merge(IMAGE, Dict("filter" => "bilinear"))))
        files = ["processor_config.json" => processor, "preprocessor_config.json" => IMAGE,
            "video_preprocessor_config.json" => VIDEO]
        with_files(files) do dir
            p = @test_logs create_processor(dir)
            @test p isa MetalheadTestProcessor
            @test p.image_processor.filter == "bilinear"
            @test p.video_processor.padding_side == "left"
            @test occursin(r"filter = \"bilinear\" +processor_config\.json", plain(p))
        end
        # in preprocessor_config.json, whose image processor it holds, over the stale class
        # of tokenizer_config.json
        image = with_class(IMAGE, "MetalheadTestProcessor")
        unported = Dict("processor_class" => "MetalheadTestUnported")
        with_files(["preprocessor_config.json" => image,
            "video_preprocessor_config.json" => VIDEO,
            "tokenizer_config.json" => unported]) do dir
            p = create_processor(dir)
            @test p isa MetalheadTestProcessor
            @test occursin(r"filter = \"bicubic\" +preprocessor_config\.json", plain(p))
        end
        # in video_preprocessor_config.json, without a preprocessor_config.json
        video = with_class(VIDEO, "MetalheadTestAliasProcessor")
        with_files(["processor_config.json" => Dict("image_processor" => IMAGE),
            "video_preprocessor_config.json" => video]) do dir
            @test create_processor(dir) isa MetalheadTestProcessor
        end
        # in tokenizer_config.json, and in config.json
        for file in ("tokenizer_config.json", "config.json")
            class = Dict("processor_class" => "MetalheadTestProcessor")
            with_files([file => class, "preprocessor_config.json" => IMAGE,
                "video_preprocessor_config.json" => VIDEO]) do dir
                @test create_processor(dir) isa MetalheadTestProcessor
            end
        end
        # by the model type of config.json
        config = Dict("architectures" => ["MetalheadTestModel"],
            "model_type" => "metalhead_test")
        with_files(["config.json" => config, "preprocessor_config.json" => IMAGE,
            "video_preprocessor_config.json" => VIDEO]) do dir
            @test create_processor(dir) isa MetalheadTestProcessor
        end
        # else the image processor of `image_processor_type`
        fast = merge(IMAGE,
            Dict("image_processor_type" => "MetalheadTestImageProcessorFast"))
        with_files(["preprocessor_config.json" => fast]) do dir
            @test create_processor(dir) isa MetalheadTestImageProcessor
        end
        # in a subfolder
        with_files(["image/preprocessor_config.json" => IMAGE]) do dir
            @test create_processor(dir; subfolder = "image").size ==
                  (; height = 6, width = 6)
            message = "has no processor files (processor_config.json, " *
                      "preprocessor_config.json, tokenizer_config.json or tokenizer.json)"
            @test_throws message create_processor(dir)
        end

        with_files(["config.json" => Dict("model_type" => "metalhead_test"),
            "video_preprocessor_config.json" => VIDEO]) do dir
            message = "has no image processor (preprocessor_config.json, or " *
                      "`image_processor` in processor_config.json)"
            @test_throws message create_processor(dir)
        end
        with_files(["tokenizer_config.json" => unported]) do dir
            message = "the processor class `MetalheadTestUnported` (`processor_class` in " *
                      "tokenizer_config.json) has not been ported to Metalhead"
            @test_throws message create_processor(dir)
        end
        image = merge(IMAGE, Dict("image_processor_type" => "MetalheadTestProcessor"))
        files = ["processor_config.json" => PROCESSOR, "preprocessor_config.json" => image,
            "video_preprocessor_config.json" => VIDEO]
        with_files(files) do dir
            message = "the image processor of preprocessor_config.json is a " *
                      "MetalheadTestProcessor, not a MetalheadTestImageProcessor"
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
        # a value that the port doesn't implement, in an override or in a file, which the
        # error names
        unsupported = "`do_center_crop = true` is not supported by " *
                      "MetalheadTestImageProcessor (supported: false)"
        config = merge(IMAGE, Dict("do_center_crop" => false))
        with_files(["preprocessor_config.json" => config]) do dir
            @test (@test_logs create_processor(dir; do_center_crop = false)) isa
                  MetalheadTestImageProcessor
            @test_throws ArgumentError(unsupported) create_processor(dir;
                do_center_crop = true)
        end
        config = merge(IMAGE, Dict("do_center_crop" => true))
        with_files(["preprocessor_config.json" => config]) do dir
            message = "$unsupported; `do_center_crop` is set in preprocessor_config.json"
            @test_throws ArgumentError(message) create_processor(dir)
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
        # `null` in a file is missing
        config = merge(IMAGE, Dict("filter" => nothing))
        with_files(["preprocessor_config.json" => config]) do dir
            ip = create_processor(dir)
            @test ip.filter == "bilinear"
            @test occursin(r"filter = \"bilinear\" +class default", plain(ip))
        end
        entries = [("fps", 3, "a.json"),
            ("padding_side", "left", "a.json"), ("padding_side", "right", "b.json"),
            ("num_frames", nothing, "b.json")]
        vp = processor_from_entries(MetalheadTestVideoProcessor, entries)
        @test vp.fps == 3 && vp.padding_side == "right"
        @test vp.provenance.fps.source == "a.json"
        @test vp.provenance.padding_side.source == "b.json"
        @test vp.provenance.num_frames.source == "class default"
        # the provenance holds the values as the constructor stored them
        ip = processor_from_entries(MetalheadTestImageProcessor,
            [("size", 5, "c.json"), ("filter", :bicubic, "c.json")])
        @test ip.provenance.size.value == (; height = 5, width = 5)
        @test ip.provenance.filter.value == "bicubic"
        message = "MetalheadTestVideoProcessor has no setting `pad`"
        @test_throws message processor_from_entries(MetalheadTestVideoProcessor, entries;
            pad = 1)
        # the error of a value names the file that sets it
        message = "`size` is an integer or a NamedTuple, got \"large\"; " *
                  "`size` is set in b.json"
        @test_throws ArgumentError(message) processor_from_entries(
            MetalheadTestImageProcessor, [("size", "large", "b.json")])
    end

    @testset "overrides" begin
        with_files(COMPOSITE) do dir
            # each to the component that has a setting of its name, NamedTuples merged into
            # nested settings
            p = create_processor(dir; size = (; height = 8), num_frames = 4,
                num_image_tokens = 2, fps = nothing)
            @test p.image_processor.size == (; height = 8, width = 6)
            @test p.video_processor.num_frames == 4 && p.video_processor.fps === nothing
            @test p.num_image_tokens == 2
            p = create_processor(dir; max_height = 5, do_center_crop = false)
            @test p.image_processor.size == (; height = 5, width = 6)
            @test_throws ["`padding_side` is a setting of several components of " *
                          "MetalheadTestProcessor (video_processor, self)",
                "set it on one of them after loading"] create_processor(dir;
                padding_side = "left")
            @test_throws ["no component of MetalheadTestProcessor has a setting " *
                          "`num_frams` (MetalheadTestVideoProcessor has " *
                          "`num_frames`)"] create_processor(dir; num_frams = 4)
            message = "no component of MetalheadTestProcessor has a setting `colour`"
            @test_throws message create_processor(dir; colour = 1)
            @test_throws ["`feature_extractor` is not used by MetalheadTestProcessor: " *
                          "audio isn't ported"] create_processor(dir; feature_extractor = 1)
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
            (; filter = "bilinear", fps = nothing, num_image_tokens = 1))
        @test routed.self == (; num_image_tokens = 1)
        @test routed.image_processor == (; filter = "bilinear")
        @test routed.video_processor == (; fps = nothing)
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
            @test occursin(r"\n  video_processor: MetalheadTestVideoProcessor\n" *
                           r"    num_frames = 16 +video_preprocessor_config\.json\n", text)
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
                "  do_center_crop = false", "  do_pad = false"]
            @test sprint(show, p.video_processor) ==
                  "MetalheadTestVideoProcessor(num_frames = 16, padding_side = \"left\", " *
                  "fps = 2.0, video_metadata = 0 entries)"
            compact = sprint(show, p)
            @test startswith(compact, "MetalheadTestProcessor(image_processor = " *
                                      "MetalheadTestImageProcessor(do_resize = true, ")
            @test endswith(compact, "; padding_side = \"right\", num_image_tokens = 6)")
        end
        # long values are cut
        vp = MetalheadTestVideoProcessor(; padding_side = "x"^100)
        @test occursin("padding_side = \"" * "x"^70 * "…, fps", sprint(show, vp))
    end

    @testset "files" begin
        with_files(["video_preprocessor_config.json" => VIDEO]) do dir
            files = files_of(dir)
            @test Metalhead.has_file(files, "video_preprocessor_config.json")
            @test !Metalhead.has_file(files, "processor_config.json")
            config = read_json(files, "video_preprocessor_config.json")
            @test config["padding_side"] == "left"
            @test read_json(files, "video_preprocessor_config.json") === config  # read once
            @test read_json(files, "processor_config.json") === nothing
            @test !Metalhead.has_file(files, "processor_config.json")
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
            videos = [[1, 2, 3], [4]]
            out = p(; images = [image, image], videos)
            @test size(out.pixel_values) == (6, 6, 3, 2)
            @test out.pixel_values_videos == [1 0; 2 0; 3 4]  # padded on the left
            check_documented_keys(MetalheadTestImageProcessor,
                [p.image_processor(image), q.image_processor([image])])
            check_documented_keys(MetalheadTestVideoProcessor, [p.video_processor(videos)])
            check_documented_keys(MetalheadTestProcessor,
                [out, q(; images = image), p(; videos)])
        end
    end
finally
    foreach(name -> delete!(Metalhead.PROCESSOR_CLASSES, name), CLASSES)
    foreach(type -> delete!(Metalhead.PROCESSOR_MODEL_TYPES, type), MODEL_TYPES)
end
