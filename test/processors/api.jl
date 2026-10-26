using Metalhead, Test
using JSON: JSON
using Metalhead: merge_nested
using Metalhead.Transformers: TokenizersBackend

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

@testset "AutoProcessor's lookup" begin
    tokenizer = sam3_json("tokenizer_config.json")
    # without a `processor_class`, a tokenizer, else an image processor
    delete!(tokenizer, "processor_class")
    with_files(["config.json" => Dict("architectures" => ["LlamaForCausalLM"],
            "model_type" => "llama"),
        "tokenizer_config.json" => tokenizer]) do dir
        @test create_processor(dir) isa TokenizersBackend
    end
    with_files(["preprocessor_config.json" => Dict("do_resize" => true)]) do dir
        @test_throws "no `image_processor_type`" create_processor(dir)
    end
    with_files(["config.json" => Dict("architectures" => ["LlamaForCausalLM"],
        "model_type" => "llama")]) do dir
        @test_throws "has no processor files" create_processor(dir)
    end
    beit = Dict("image_processor_type" => "BeitImageProcessorFast")
    with_files(["preprocessor_config.json" => beit]) do dir
        @test_throws "the processor class `BeitImageProcessorFast` (`image_processor_type` in preprocessor_config.json) has not been ported" create_processor(dir)
    end
end

