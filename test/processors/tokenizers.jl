using Metalhead, Test
using JSON: JSON
using SafeTensors: SafeTensors
using Metalhead: ProcessorFiles, model_source, load_tokenizer, encode_ids, token_id,
                 pad_token_id, pad_ids, tokenizer_of, load_processor, AbstractProcessor,
                 AbstractImageProcessor, @processor, TextProcessor, DeferredTokenizer,
                 load_tokenizer_or_defer, resolve_tokenizer

include(joinpath(@__DIR__, "..", "processors_setup.jl"))

const REFERENCES = joinpath(pkgdir(Metalhead), "test", "references")

files(name; subfolder = "") = ProcessorFiles(model_source(name, nothing, subfolder))
reference_file(name) = joinpath(REFERENCES, name * ".safetensors")
reference(name) = Metalhead.Checkpoint(reference_file(name))
metadata(name) = SafeTensors.deserialize(reference_file(name)).metadata

# A directory with the files `files` (name => Dict, or a string)
function with_files(f, files)
    mktempdir() do dir
        for (name, content) in files
            write(joinpath(dir, name), content isa String ? content : JSON.json(content))
        end
        return f(dir)
    end
end

@processor struct MetalheadTestImageProcessor <: AbstractImageProcessor
    size::Int = 4
    provenance::NamedTuple = (;)
end

# A processor that combines an image processor and a tokenizer
@processor (image_processor, tokenizer) struct MetalheadTestProcessor <: AbstractProcessor
    image_processor::Union{Nothing, MetalheadTestImageProcessor} = nothing
    tokenizer::Any = nothing
    max_length::Int = 8
    provenance::NamedTuple = (;)
end

struct MetalheadTestTokenizer end

const COMPOSITE = ["preprocessor_config.json" => Dict("size" => 6),
    "processor_config.json" => Dict("max_length" => 16),
    "tokenizer_config.json" => Dict("pad_token" => "<pad>")]

@testset "pad_ids" begin
    out = pad_ids([[5, 6, 7], [8]], 0)
    @test out.input_ids == [6 9; 7 1; 8 1]
    @test out.attention_mask == [1 1; 1 0; 1 0]
    out = pad_ids([[5, 6]], 3; max_length = 4)
    @test out.input_ids == reshape([6, 7, 4, 4], 4, 1)
    @test out.attention_mask == reshape([1, 1, 0, 0], 4, 1)
    # a tokenizer without a padding token pads nothing
    @test pad_ids([[5], [6]], nothing) == (; input_ids = [6 7], attention_mask = [1 1])
    @test_throws "has no padding token, to pad a text of 1 tokens to 2" pad_ids(
        [[5, 6], [7]], nothing)
    @test_throws "a text has 2 tokens, more than the 1" pad_ids([[5, 6]], 0; max_length = 1)
    @test size(pad_ids(Vector{Int}[], 0).input_ids) == (0, 0)
end

@testset "a processor with a tokenizer" begin
    T = MetalheadTestProcessor
    @test Metalhead.component_classes(T) ==
          (; image_processor = MetalheadTestImageProcessor, tokenizer = Any)
    # without tokenizer.json, no tokenizer, with HuggingFaceTokenizers or without it
    with_files(COMPOSITE) do dir
        p = load_processor(T, files(dir))
        @test p.tokenizer === nothing && p.image_processor.size == 6 && p.max_length == 16
        @test_throws ["MetalheadTestProcessor has no tokenizer to encode text",
            "`using HuggingFaceTokenizers` before `create_processor`", "tokenizer.json"] tokenizer_of(p)
        # overrides go to the components that have settings, never to the tokenizer
        p = load_processor(T, files(dir); size = 3, max_length = 4)
        @test p.image_processor.size == 3 && p.max_length == 4
        @test_throws "no component of MetalheadTestProcessor has a setting `pad_token`" load_processor(
            T, files(dir); pad_token = "<eos>")
        @test_throws "(MetalheadTestProcessor has `max_length`)" load_processor(T,
            files(dir); max_lenght = 4)
        @test sprint(show, p) ==
              "MetalheadTestProcessor(image_processor = MetalheadTestImageProcessor(size = 3), tokenizer = nothing; max_length = 4)"
    end
    # a tokenizer is shown by its type
    p = T(MetalheadTestImageProcessor(), MetalheadTestTokenizer())
    @test tokenizer_of(p) isa MetalheadTestTokenizer
    @test occursin("tokenizer = MetalheadTestTokenizer;", sprint(show, p))
    plain = sprint(show, MIME("text/plain"), p)
    @test endswith(plain, "\n  tokenizer: MetalheadTestTokenizer")
end

# A tokenizer of words, whose ids are those of `vocab`, after the id 9 of a special token
struct WordTokenizer
    pad::Union{Nothing, Int}
end
const VOCAB = Dict("hello" => 3, "world" => 4, "again" => 5)
function Metalhead.encode_ids(::WordTokenizer, text::AbstractString;
        add_special_tokens::Bool = true)
    ids = [VOCAB[w] for w in split(text)]
    return add_special_tokens ? [9; ids] : ids
end
Metalhead.pad_token_id(t::WordTokenizer) = t.pad

@testset "TextProcessor" begin
    tok = TextProcessor(WordTokenizer(7))
    # one text gives vectors, as the model takes them
    @test tok("hello world") == (; input_ids = [10, 4, 5], attention_mask = [1, 1, 1])
    @test tok(; text = "hello world") == tok("hello world")
    @test tok("hello world"; add_special_tokens = false).input_ids == [4, 5]
    # a batch, padded on the right with the padding token
    out = tok(["hello", "hello world again"])
    @test out.input_ids == [10 10; 4 4; 8 5; 8 6]
    @test out.attention_mask == [1 1; 1 1; 0 1; 0 1]
    check_documented_keys(TextProcessor, [out])
    # without a padding token, with the token index 1, which the attention mask leaves out
    @test TextProcessor(WordTokenizer(nothing))(["hello", "hello world"]).input_ids ==
          [10 10; 4 4; 1 5]
    @test_throws "a string or a vector of strings" tok(1)
    @test_throws "a string or a vector of strings" tok(["hello", 1])
    @test_throws "TextProcessor has no tokenizer to encode text" TextProcessor()("hello")
    @test sprint(show, tok) == "TextProcessor(tokenizer = WordTokenizer)"
    @test sprint(show, MIME("text/plain"), tok) == "TextProcessor\n  tokenizer: WordTokenizer"
    @test_throws "no settings to override (got `padding_side`)" load_processor(TextProcessor,
        files(mktempdir()); padding_side = "left")
end

# With METALHEAD_TEST_NETWORK, another test of this worker may have loaded
# HuggingFaceTokenizers
if Base.get_extension(Metalhead, :MetalheadHuggingFaceTokenizersExt) === nothing
    @testset "without HuggingFaceTokenizers" begin
        mktempdir() do dir
            write(joinpath(dir, "tokenizer.json"), "{}")
            @test load_tokenizer(files(dir)) === nothing
            @test_throws "run `using HuggingFaceTokenizers` before `create_processor`" load_processor(
                TextProcessor, files(dir))
            # a pipeline keeps its tokenizer.json, which loads with HuggingFaceTokenizers
            deferred = load_tokenizer_or_defer(files(dir))
            @test deferred == DeferredTokenizer(joinpath(dir, "tokenizer.json"), nothing)
            @test resolve_tokenizer(deferred) === nothing
            write(joinpath(dir, "tokenizer_config.json"), JSON.json(Dict("pad_token" => "<pad>")))
            @test load_tokenizer_or_defer(files(dir)).pad == "<pad>"
        end
        mktempdir() do dir
            @test load_tokenizer_or_defer(files(dir)) === nothing
        end
        @test resolve_tokenizer(WordTokenizer(1)) == WordTokenizer(1)
        @test_throws MethodError encode_ids("tokenizer", "text")
    end
end

# HuggingFaceTokenizers.jl, which METALHEAD_TEST_NETWORK adds to the test environment
# (runtests.jl), runs Python's tokenizers through PythonCall
const TEST_NETWORK = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))
if TEST_NETWORK
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers, Tokenizer
    @eval using Flux: Flux
    @eval using Functors: Functors

    # Each language model's reference holds its prompts as transformers tokenizes them, cut
    # to the length of the shortest, and the tokenizer_config.json of each has these padding
    # tokens
    @testset "language models: $repo" for (repo, pad) in [
        "HuggingFaceTB/SmolLM2-135M" => nothing,
        "Qwen/Qwen3-0.6B" => "<|endoftext|>" => 151643,
        "Qwen/Qwen3-30B-A3B" => "<|endoftext|>" => 151643,
        "unsloth/Llama-3.2-1B" => "<|finetune_right_pad_id|>" => 128004,
        "unsloth/gemma-3-270m" => "<pad>" => 0]
        proc = @test_logs create_processor(repo)
        @test proc isa TextProcessor
        tok = proc.tokenizer
        @test tok isa Tokenizer
        @test load_tokenizer(files(repo)) === tok
        name = replace(repo, "/" => "--")
        ids = reference(name)["input_ids"]
        prompts = JSON.parse(metadata(name)["prompts"])
        n = size(ids, 1)
        for (b, prompt) in enumerate(prompts)
            @test encode_ids(tok, prompt)[1:n] == ids[:, b]
            @test HuggingFaceTokenizers.encode(tok, prompt).ids[1:n] == ids[:, b]
            # the processor gives token indices, from 1, and decodes them
            inputs = proc(prompt)
            @test inputs.input_ids == encode_ids(tok, prompt) .+ 1
            @test all(==(1), inputs.attention_mask)
            @test HuggingFaceTokenizers.decode(proc, inputs.input_ids) ==
                  HuggingFaceTokenizers.decode(tok, encode_ids(tok, prompt))
        end
        @test HuggingFaceTokenizers.decode(proc, ids[end, 1] + 1) ==
              HuggingFaceTokenizers.decode(tok, [ids[end, 1]])
        batch = proc(prompts)
        lengths = [length(encode_ids(tok, p)) for p in prompts]
        @test batch.attention_mask == [Int(i <= l) for i in 1:maximum(lengths), l in lengths]
        padding = something(pad_token_id(tok), 0) + 1
        @test batch.input_ids == [i <= l ? proc(p).input_ids[i] : padding
                                  for i in 1:maximum(lengths), (p, l) in zip(prompts, lengths)]
        if pad === nothing
            @test pad_token_id(tok) === nothing
            @test_throws "no padding token" pad_ids(encode_ids.((tok,), prompts),
                pad_token_id(tok))
        else
            @test pad_token_id(tok) == token_id(tok, first(pad)) == last(pad)
        end
        @test token_id(tok, "not a single token") === nothing
    end

    @testset "special tokens" begin
        tok = load_tokenizer(files("unsloth/Llama-3.2-1B"))
        ids = encode_ids(tok, "The capital of France is")
        @test first(ids) == token_id(tok, "<|begin_of_text|>") == 128000
        @test encode_ids(tok, "The capital of France is"; add_special_tokens = false) ==
              ids[2:end]
        @test HuggingFaceTokenizers.decode(tok, ids; skip_special_tokens = true) ==
              "The capital of France is"
    end

    # SAM 3's CLIP tokenizer, as porting/dump_processor.py called transformers': the token ids
    # of each text, and batches padded on the right to the longest and to 12 tokens
    @testset "SAM 3" begin
        file = joinpath(REFERENCES, "processors", "sam3.safetensors")
        ref = Metalhead.Checkpoint(file)
        texts = JSON.parse(SafeTensors.deserialize(file).metadata["tokenizer/texts"])
        tok = load_tokenizer(files("unsloth/sam3.1"))
        for (i, text) in enumerate(texts)
            @test encode_ids(tok, text) == ref["tokenizer/ids/$i"]
            @test encode_ids(tok, text; add_special_tokens = false) ==
                  ref["tokenizer/content_ids/$i"]
        end
        @test pad_token_id(tok) == token_id(tok, "<|endoftext|>") == 49407
        batch = encode_ids.((tok,), texts[1:2])
        for (name, max_length) in ["longest" => nothing, "max_length" => 12]
            out = pad_ids(batch, pad_token_id(tok); max_length)
            @test out.input_ids == ref["tokenizer/$name/input_ids"] .+ 1
            @test out.attention_mask == ref["tokenizer/$name/attention_mask"]
        end
    end

    # A tokenizer.json of words, with the padding `padding`
    function tokenizer_json(padding = nothing)
        vocab = Dict("[UNK]" => 0, "[PAD]" => 1, "<pad>" => 2, "hello" => 3, "world" => 4)
        model = Dict("type" => "WordLevel", "vocab" => vocab, "unk_token" => "[UNK]")
        return Dict("version" => "1.0", "truncation" => nothing, "padding" => padding,
            "added_tokens" => [], "normalizer" => nothing,
            "pre_tokenizer" => Dict("type" => "Whitespace"), "post_processor" => nothing,
            "decoder" => nothing, "model" => model)
    end

    # The padding of tokenizer.json, if it has one, wins over the pad_token of
    # tokenizer_config.json
    @testset "padding token" begin
        padding = Dict("strategy" => "BatchLongest", "direction" => "Right",
            "pad_to_multiple_of" => nothing, "pad_id" => 1, "pad_type_id" => 0,
            "pad_token" => "[PAD]")
        function load(padding, config)
            dir = mktempdir()
            write(joinpath(dir, "tokenizer.json"), JSON.json(tokenizer_json(padding)))
            config === nothing ||
                write(joinpath(dir, "tokenizer_config.json"), JSON.json(config))
            return load_tokenizer(files(dir))
        end
        @test pad_token_id(load(nothing, Dict("pad_token" => "<pad>"))) == 2
        content = Dict("content" => "<pad>", "special" => true)
        @test pad_token_id(load(nothing, Dict("pad_token" => content))) == 2
        @test pad_token_id(load(padding, Dict("pad_token" => "<pad>"))) == 1
        @test pad_token_id(load(nothing, Dict("pad_token" => nothing))) === nothing
        @test pad_token_id(load(nothing, nothing)) === nothing
        # a pipeline loaded before HuggingFaceTokenizers keeps the file of its tokenizer
        dir = mktempdir()
        write(joinpath(dir, "tokenizer.json"), JSON.json(tokenizer_json()))
        deferred = DeferredTokenizer(joinpath(dir, "tokenizer.json"), "<pad>")
        @test resolve_tokenizer(deferred) isa Tokenizer
        @test pad_token_id(resolve_tokenizer(deferred)) == 2
        @test load_tokenizer_or_defer(files(dir)) isa Tokenizer
        tok = @test_logs (:warn, r"not in the vocabulary") load(nothing,
            Dict("pad_token" => "<absent>"))
        @test pad_token_id(tok) === nothing
        # a tokenizer.json that pads to a fixed length: the ids are those of the text
        fixed = merge(padding, Dict("strategy" => Dict("Fixed" => 4)))
        @test encode_ids(load(fixed, nothing), "hello world") == [3, 4]
    end

    @testset "without tokenizer.json" begin
        with_files(["tokenizer_config.json" => Dict("pad_token" => "<pad>")]) do dir
            @test load_tokenizer(files(dir)) === nothing
            @test_throws "has no tokenizer.json, the file of its tokenizer" create_processor(
                dir)
        end
    end

    @testset "a processor with a tokenizer, loaded" begin
        with_files([COMPOSITE; "tokenizer.json" => JSON.json(tokenizer_json())]) do dir
            p = load_processor(MetalheadTestProcessor, files(dir); size = 3)
            @test p.tokenizer isa Tokenizer && p.image_processor.size == 3
            @test tokenizer_of(p) === p.tokenizer
            @test pad_token_id(p.tokenizer) == 2
            # by its type, not its Python repr, which holds the vocabulary
            @test occursin("tokenizer = Tokenizer;", sprint(show, p))
            @test endswith(sprint(show, MIME("text/plain"), p), "\n  tokenizer: Tokenizer")
            @test !occursin("WordLevel", sprint(show, MIME("text/plain"), p))
        end
    end

    @testset "a leaf" begin
        tok = load_tokenizer(files("Qwen/Qwen3-0.6B"))
        @test Functors.isleaf(tok)
        x = Flux.f16((; weight = ones(Float32, 2), tok))
        @test x.tok === tok && eltype(x.weight) == Float16
    end

    # the Qwen3 tokenizer of FLUX.2 [klein], as its pipeline calls it
    @testset "FLUX.2 [klein]" begin
        D = Metalhead.Diffusers
        repo = "black-forest-labs/FLUX.2-klein-4B"
        name = replace(repo, "/" => "--")
        ref = reference(name)
        prompt = metadata(name)["prompt"]
        tok = (@test_logs create_processor(repo; subfolder = "tokenizer")).tokenizer
        @test tok isa Tokenizer
        # the tokenizer of the pipeline, and the inputs of its `encode_prompt`
        @test D.pipeline_tokenizer(Metalhead.subsource(model_source(repo, nothing),
            "tokenizer")) === tok
        out = D.qwen3_prompt_inputs(tok, prompt)
        @test out.input_ids == ref["input_ids"] .+ 1
        @test out.attention_mask == ref["attention_mask"]
        short = D.qwen3_prompt_inputs(tok, [prompt, "a"]; max_sequence_length = 8)
        @test size(short.input_ids) == (8, 2) && all(==(1), short.attention_mask[:, 1])
        @test short.input_ids[:, 1] == ref["input_ids"][1:8] .+ 1
    end

    # the Qwen3 tokenizer of Z-Image-Turbo, whose pipeline formats prompts with thinking
    # enabled
    @testset "Z-Image-Turbo" begin
        D = Metalhead.Diffusers
        repo = "Tongyi-MAI/Z-Image-Turbo"
        name = replace(repo, "/" => "--")
        ref = reference(name)
        tok = (@test_logs create_processor(repo; subfolder = "tokenizer")).tokenizer
        @test tok isa Tokenizer
        out = D.z_image_prompt_inputs(tok, metadata(name)["prompt"])
        @test out.input_ids == ref["input_ids"] .+ 1
        @test out.attention_mask == ref["attention_mask"]
    end
end
