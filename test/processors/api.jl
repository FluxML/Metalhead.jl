using Metalhead, Test
using JSON: JSON
using Metalhead: merge_nested, image_array, image_batch, ProcessorFiles
using Metalhead.Transformers: Sam3Processor, Sam3ImageProcessor, TokenizersBackend

const SAM3 = joinpath(pkgdir(Metalhead), "test", "fixtures", "processors",
    "unsloth--sam3.1")

# A directory with the JSON files `files` (name => Dict)
function with_files(f, files)
    mktempdir() do dir
        for (name, content) in files
            write(joinpath(dir, name), JSON.json(content))
        end
        return f(dir)
    end
end

sam3_json(name) = JSON.parsefile(joinpath(SAM3, name))

@testset "merge_nested" begin
    base = (; a = 1, b = (; c = 2, d = 3))
    @test merge_nested(base, (; b = (; d = 4))) == (; a = 1, b = (; c = 2, d = 4))
    @test merge_nested(base, (; b = 5, e = 6)) == (; a = 1, b = 5, e = 6)
    @test merge_nested(base, (; b = (; f = (; g = 1)))) ==
          (; a = 1, b = (; c = 2, d = 3, f = (; g = 1)))
    @test merge_nested(1, (; x = 2)) == (; x = 2)
    @test merge_nested(base, (;)) == base
end

@testset "images" begin
    x = rand(UInt8, 4, 3, 3)
    @test image_array(x) === x
    @test size(image_array(rand(Float32, 4, 3))) == (4, 3, 1)
    @test_throws "(width, height, channels)" image_array("image")
    @test image_batch(x) == [x]
    @test image_batch([x, x]) == [x, x]
    @test image_batch(cat(x, x; dims = 4)) == [x, x]
    @test size.(image_batch(rand(4, 3))) == [(4, 3, 1)]
    @test Metalhead.convert_to_rgb(image_array(x[:, :, 1])) == repeat(x[:, :, 1:1], 1, 1, 3)
    @test Metalhead.convert_to_rgb(cat(x, x[:, :, 1:1]; dims = 3)) == x  # alpha dropped
    @test_throws "1 to 4 channels" Metalhead.convert_to_rgb(rand(UInt8, 2, 2, 5))
end

@testset "AutoProcessor's lookup" begin
    tokenizer = sam3_json("tokenizer_config.json")
    image_processor = sam3_json("processor_config.json")["image_processor"]
    # the processor of processor_config.json wins over the stale `processor_class` of
    # tokenizer_config.json
    @test create_processor(SAM3) isa Sam3Processor
    # a processor named in preprocessor_config.json, whose image processor it holds
    with_files(["preprocessor_config.json" => merge(image_processor,
            Dict("processor_class" => "Sam3Processor")),
        "tokenizer_config.json" => tokenizer]) do dir
        proc = create_processor(dir)
        @test proc isa Sam3Processor
        @test occursin(r"do_resize = true +preprocessor_config\.json",
            sprint(show, MIME("text/plain"), proc))
    end
    # by the model type of config.json, without a `processor_class`
    delete!(tokenizer, "processor_class")
    config = Dict("architectures" => ["Sam3Model"], "model_type" => "sam3")
    with_files(["config.json" => config, "preprocessor_config.json" => image_processor,
        "tokenizer_config.json" => tokenizer]) do dir
        @test create_processor(dir) isa Sam3Processor
    end
    # else a tokenizer, else an image processor
    with_files(["config.json" => Dict("architectures" => ["LlamaForCausalLM"],
            "model_type" => "llama"),
        "tokenizer_config.json" => tokenizer]) do dir
        @test create_processor(dir) isa TokenizersBackend
    end
    with_files(["preprocessor_config.json" => image_processor]) do dir
        ip = create_processor(dir; size = (; height = 8, width = 8))
        @test ip isa Sam3ImageProcessor
        @test ip.size == (; height = 8, width = 8)
        @test_throws "Sam3ImageProcessor has no setting `point_pad_value`" create_processor(dir;
            point_pad_value = 1)
    end
    with_files(["preprocessor_config.json" => Dict("do_resize" => true)]) do dir
        @test_throws "no `image_processor_type`" create_processor(dir)
    end
    with_files(["config.json" => config]) do dir
        @test_throws "has no image processor" create_processor(dir)
    end
    with_files(["config.json" => Dict("architectures" => ["LlamaForCausalLM"],
        "model_type" => "llama")]) do dir
        @test_throws "has no processor files" create_processor(dir)
    end
    with_files(["preprocessor_config.json" => merge(image_processor,
        Dict("image_processor_type" => "BeitImageProcessorFast"))]) do dir
        @test_throws "the processor class `BeitImageProcessorFast` (`image_processor_type` in preprocessor_config.json) has not been ported" create_processor(dir)
    end
    @test_throws "already registered" Metalhead.register_processor!("Sam3Processor",
        Sam3ImageProcessor)
    @test Metalhead.processor_type("Sam3VideoProcessor", "") === Sam3Processor
    @test Metalhead.processor_type("Sam3ImageProcessorPil", "") === Sam3ImageProcessor
end

# A library whose config.json carries the preprocessing supplies its processor through
# `instantiate_processor`, as timm's `pretrained_cfg` does
struct TestFormat <: Metalhead.ConfigFormat end
Metalhead.architecture_of(::TestFormat, config) = get(config, "test_architecture", nothing)
function Metalhead.instantiate_processor(::TestFormat, files, config; kwargs...)
    entries = Metalhead.json_entries(config["preprocessing"], "config.json")
    return Metalhead.processor_from_entries(Sam3ImageProcessor, entries; kwargs...)
end

@testset "the processor of a config.json" begin
    spec = Metalhead.ArchSpec(TestFormat(), "TestProcessorArchitecture", () -> nothing)
    Metalhead.register!(spec)
    try
        config = Dict("test_architecture" => "TestProcessorArchitecture",
            "preprocessing" => Dict("resample" => 3))
        with_files(["config.json" => config]) do dir
            ip = create_processor(dir; do_normalize = false)
            @test ip isa Sam3ImageProcessor
            @test ip.resample == 3 && !ip.do_normalize
            text = sprint(show, MIME("text/plain"), ip)
            @test occursin(r"resample = 3 \(bicubic\) +config\.json", text)
            @test occursin(r"do_normalize = false +override \(class default: true\)", text)
        end
        @test Metalhead.instantiate_processor(Metalhead.Transformers.TRANSFORMERS,
            ProcessorFiles(Metalhead.model_source(SAM3, nothing)), Dict()) === nothing
    finally
        delete!(Metalhead.REGISTRY, spec.name)
    end
end
