using Metalhead, Test
using JSON: JSON
using Metalhead: merge_nested, image_array, image_batch

# A directory with the JSON files `files` (name => Dict)
function with_files(f, files)
    mktempdir() do dir
        for (name, content) in files
            write(joinpath(dir, name), JSON.json(content))
        end
        return f(dir)
    end
end

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
    with_files(["preprocessor_config.json" => Dict("do_resize" => true)]) do dir
        @test_throws "no `image_processor_type`" create_processor(dir)
    end
    llama = Dict("architectures" => ["LlamaForCausalLM"], "model_type" => "llama")
    with_files(["config.json" => llama]) do dir
        @test_throws "has no processor files" create_processor(dir)
    end
    # without a `processor_class`, the tokenizer, before an image processor; with
    # METALHEAD_TEST_NETWORK, another test of this worker may have loaded
    # HuggingFaceTokenizers
    tokenizer = Dict("pad_token" => "<pad>")
    with_files(["config.json" => llama, "tokenizer_config.json" => tokenizer,
        "preprocessor_config.json" => Dict("do_resize" => true)]) do dir
        @test_throws "a tokenizer, which has no settings to override (got `padding_side`)" create_processor(
            dir; padding_side = "left")
        ext = Base.get_extension(Metalhead, :MetalheadHuggingFaceTokenizersExt)
        @test_throws (ext === nothing ?
                      "run `using HuggingFaceTokenizers` before `create_processor`" :
                      "has no tokenizer.json") create_processor(dir)
    end
    beit = Dict("image_processor_type" => "BeitImageProcessorFast")
    with_files(["preprocessor_config.json" => beit]) do dir
        @test_throws "the processor class `BeitImageProcessorFast` (`image_processor_type` in preprocessor_config.json) has not been ported" create_processor(dir)
    end
end
