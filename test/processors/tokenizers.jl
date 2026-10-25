using Metalhead, Test
using JSON: JSON
using SafeTensors: SafeTensors
using Metalhead.Transformers: TokenizersBackend, decode
const T = Metalhead.Transformers

# Written by porting/dump_processor.py: the CLIP tokenizer of SAM 3 in transformers
const SAM3 = joinpath(pkgdir(Metalhead), "test", "fixtures", "processors",
    "unsloth--sam3.1")
const FILE = joinpath(pkgdir(Metalhead), "test", "references", "processors",
    "sam3.safetensors")
const META = SafeTensors.deserialize(FILE).metadata
const REF = Metalhead.Checkpoint(FILE)
const TEXTS = JSON.parse(META["tokenizer/texts"])
const CALLS = ["longest", "max_length", "left", "truncate", "truncate_only",
    "model_max_length", "no_special"]

ids(i) = REF["tokenizer/ids/$i"]
python_bool(b) = b ? "True" : "False"
content_ids(i) = REF["tokenizer/content_ids/$i"]

# A backend that knows the texts of the reference: a tokenizer backend is any type with
# methods of `encode_ids` and `decode_ids`.
struct ReferenceBackend end
function T.encode_ids(::ReferenceBackend, text::AbstractString; add_special_tokens = true)
    i = findfirst(==(text), TEXTS)
    return add_special_tokens ? ids(i) : content_ids(i)
end
function T.decode_ids(::ReferenceBackend, ids::AbstractVector{<:Integer};
        skip_special_tokens = false)
    decoded = JSON.parse(META["tokenizer/decode"])
    @assert ids == REF["tokenizer/ids/$(decoded["ids"])"]
    return decoded[python_bool(skip_special_tokens)]["raw"]
end

call_kwargs(m) = (; (Symbol(k) => v isa String ? Symbol(v) : v
                     for (k, v) in m if k != "text")...)

function with_tokenizer_files(f, files)
    mktempdir() do dir
        for (name, content) in files
            write(joinpath(dir, name), content isa String ? content : JSON.json(content))
        end
        return f(dir)
    end
end

@testset "token ids" begin
    tok = create_processor(SAM3).tokenizer
    out = tok(ids(1))
    @test out == (; input_ids = ids(1) .+ 1, attention_mask = ones(Int, 4))
    for name in ("longest", "max_length", "left")
        m = JSON.parse(META["tokenizer/$name"])
        out = tok([ids(i) for i in m["text"]]; call_kwargs(m)...)
        @test out.input_ids == REF["tokenizer/$name/input_ids"] .+ 1
        @test out.attention_mask == REF["tokenizer/$name/attention_mask"]
    end
    @test tok([ids(1), ids(2)]).input_ids == REF["tokenizer/longest/input_ids"] .+ 1
    @test tok([ids(1), ids(1)]; padding = :max_length).input_ids ==
          [ids(1) .+ 1; fill(49408, 28);; ids(1) .+ 1; fill(49408, 28)]  # model_max_length
    @test_throws "pad them" tok([ids(1), ids(2)]; padding = false)
    @test_throws "longer than `max_length = 3`" tok(ids(1); truncation = true, max_length = 3)
    @test_throws "padding_side" tok([ids(1), ids(2)]; padding_side = "middle")
    @test_throws "`padding = :maximum`" tok(ids(1); padding = :maximum)
    if Base.get_extension(Metalhead, :MetalheadHuggingFaceTokenizersExt) === nothing
        @test_throws "using HuggingFaceTokenizers" tok("a cat")
        @test_throws "using HuggingFaceTokenizers" decode(tok, ids(1) .+ 1)
    end
end

@testset "a backend" begin
    proc = create_processor(SAM3; tokenizer = ReferenceBackend())
    tok = proc.tokenizer
    @test tok.tokenizer isa ReferenceBackend
    for name in CALLS
        m = JSON.parse(META["tokenizer/$name"])
        out = tok(TEXTS[m["text"]]; call_kwargs(m)...)
        @testset "$name" begin
            @test out.input_ids == REF["tokenizer/$name/input_ids"] .+ 1
            @test out.attention_mask == REF["tokenizer/$name/attention_mask"]
        end
    end
    @test tok(TEXTS[1]) == (; input_ids = ids(1) .+ 1, attention_mask = ones(Int, 4))
    long = TEXTS[3]
    @test tok(long; max_length = 6).input_ids == [ids(3)[1:5]; 49407] .+ 1
    left = TokenizersBackend(; Metalhead.settings(tok)..., truncation_side = "left")
    @test left(long; max_length = 6).input_ids == [49406; ids(3)[(end - 4):end]] .+ 1
    @test_throws "special tokens" tok(long; max_length = 1)
    # the processor encodes its prompts with the backend
    sam = proc(; text = TEXTS[1:2])
    @test sam.input_ids == [ids(1) .+ 1; fill(49408, 28);; ids(2) .+ 1;
                            fill(49408, 32 - length(ids(2)))]
    decoded = JSON.parse(META["tokenizer/decode"])
    for skip in (false, true)
        @test decode(tok, ids(2) .+ 1; skip_special_tokens = skip) ==
              decoded[python_bool(skip)]["text"]
    end
    @test decode(proc, ids(2) .+ 1) == decoded["False"]["text"]
    @test decode(tok, [ids(2) ids(2)] .+ 1) == fill(decoded["False"]["text"], 2)
end

@testset "tokenizer_config.json" begin
    added(tokens...) = Dict(string(id) => Dict("content" => t, "special" => true)
                            for (id, t) in tokens)
    # as transformers 5 reads it: `added_tokens_decoder`, then special_tokens_map.json only
    # without it; `additional_special_tokens` when `extra_special_tokens` is empty; named
    # special tokens of a model in `extra_special_tokens` or `*_token` keys
    config = Dict("tokenizer_class" => "Qwen2TokenizerFast", "pad_token" => "<pad>",
        "eos_token" => Dict("content" => "<eos>", "__type" => "AddedToken"),
        "bos_token" => nothing, "added_tokens_decoder" => added(0 => "<pad>", 7 => "<eos>"),
        "extra_special_tokens" => Dict("image_token" => "<img>"),
        "boi_token" => "<boi>", "additional_special_tokens" => ["<a>", "<b>"],
        "model_max_length" => big"1000000000000000019884624838656",
        "padding_side" => "left", "chat_template" => "{{ messages }}",
        "add_bos_token" => true, "do_lower_case" => true)
    map = Dict("pad_token" => "<other>", "unk_token" => "<unk>")
    with_tokenizer_files(["tokenizer_config.json" => config,
        "special_tokens_map.json" => map]) do dir
        tok = @test_logs create_processor(dir)
        @test tok isa TokenizersBackend
        @test tok.tokenizer_class == "Qwen2Tokenizer"
        @test tok.pad_token == "<pad>" && tok.eos_token == "<eos>"
        @test tok.bos_token === nothing && tok.unk_token === nothing
        @test tok.model_specific_special_tokens == (; boi_token = "<boi>", image_token = "<img>")
        @test tok.extra_special_tokens == String[]
        @test tok.added_tokens_decoder == Dict(0 => "<pad>", 7 => "<eos>")
        @test tok.model_max_length === nothing
        @test tok([[5, 6], [5]]).input_ids == [6 1; 7 6]  # padded on the left
        @test tok([5, 6]; padding = :max_length) == (; input_ids = [6, 7],
            attention_mask = [1, 1])  # no limit: no padding
    end
    legacy = Dict("pad_token" => "<pad>", "extra_special_tokens" => Dict(),
        "additional_special_tokens" => ["<a>"], "max_len" => 9)
    map = Dict("pad_token" => Dict("content" => "<other>"), "unk_token" => "<unk>",
        "extra_special_tokens" => ["<b>"], "additional_special_tokens" => ["<c>"])
    with_tokenizer_files(["tokenizer_config.json" => legacy,
        "special_tokens_map.json" => map, "added_tokens.json" => Dict("<other>" => 3)]) do dir
        tok = @test_logs create_processor(dir)
        @test tok.pad_token == "<other>" && tok.unk_token == "<unk>"
        @test tok.extra_special_tokens == ["<a>", "<b>"]
        @test tok.added_tokens_decoder == Dict(3 => "<other>")
        @test tok.model_max_length == 9
        @test tok([[1], [1, 2]]).input_ids == [2 2; 4 3]
        text = sprint(show, MIME("text/plain"), tok)
        @test occursin(r"pad_token = \"<other>\" +special_tokens_map\.json", text)
        @test occursin(r"model_max_length = 9 +tokenizer_config\.json \(`max_len`\)", text)
    end
    with_tokenizer_files(["tokenizer_config.json" => Dict("eos_token" => "</s>")]) do dir
        tok = create_processor(dir)
        @test tok([[1], [1]]).input_ids == [2 2]
        @test_throws "no padding token" tok([[1], [1, 2]])
    end
    unknown = Dict("tokenizer_test_key" => 1)
    with_tokenizer_files(["tokenizer_config.json" => unknown]) do dir
        @test_logs (:warn,
            "`tokenizer_test_key` in tokenizer_config.json is not a setting of TokenizersBackend; it is ignored") create_processor(dir)
        @test_throws "TokenizersBackend has no setting `padding_sid` (TokenizersBackend has `padding_side`)" create_processor(dir;
            padding_sid = "left")
        @test_throws "`chat_template` is not used by TokenizersBackend" create_processor(dir;
            chat_template = "")
        @test_throws "must be \"right\" or \"left\"" create_processor(dir;
            truncation_side = "middle")
    end
    for (key, value) in ("split_special_tokens" => true, "padding_side" => "middle",
        "model_input_names" => ["input_ids", "token_type_ids"])
        with_tokenizer_files(["tokenizer_config.json" => Dict(key => value)]) do dir
            @test_throws key create_processor(dir)
        end
    end
    with_tokenizer_files(["tokenizer_config.json" => Dict(
        "model_input_names" => ["input_ids"])]) do dir
        @test keys(create_processor(dir)([1, 2])) == (:input_ids,)
    end
end

# Strings through HuggingFaceTokenizers.jl, which METALHEAD_TEST_NETWORK adds to the test
# environment (runtests.jl): it wraps the Python tokenizers through PythonCall
const TEST_NETWORK = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))
if TEST_NETWORK
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers
    @testset "HuggingFaceTokenizers" begin
        proc = create_processor("unsloth/sam3.1")
        tok = proc.tokenizer
        @test tok.tokenizer === nothing
        for (i, text) in enumerate(TEXTS)
            @test tok(text).input_ids == ids(i) .+ 1
            @test tok(text; add_special_tokens = false).input_ids == content_ids(i) .+ 1
        end
        for name in CALLS
            m = JSON.parse(META["tokenizer/$name"])
            out = tok(TEXTS[m["text"]]; call_kwargs(m)...)
            @test out.input_ids == REF["tokenizer/$name/input_ids"] .+ 1
            @test out.attention_mask == REF["tokenizer/$name/attention_mask"]
        end
        decoded = JSON.parse(META["tokenizer/decode"])
        for skip in (false, true)
            @test decode(tok, ids(2) .+ 1; skip_special_tokens = skip) ==
                  decoded[python_bool(skip)]["text"]
        end

        # the Qwen3 tokenizer of FLUX.2 [klein], as its pipeline calls it
        D = Metalhead.Diffusers
        repo = "black-forest-labs/FLUX.2-klein-4B"
        flux = Metalhead.Checkpoint(joinpath(pkgdir(Metalhead), "test", "references",
            "black-forest-labs--FLUX.2-klein-4B.safetensors"))
        prompt = SafeTensors.deserialize(joinpath(pkgdir(Metalhead), "test", "references",
            "black-forest-labs--FLUX.2-klein-4B.safetensors")).metadata["prompt"]
        qwen = @test_logs create_processor(repo; subfolder = "tokenizer")
        @test qwen isa TokenizersBackend && qwen.tokenizer_class == "Qwen2Tokenizer"
        out = qwen([D.chat_prompt(prompt)]; padding = :max_length, truncation = true,
            max_length = 512)
        @test out.input_ids == flux["input_ids"] .+ 1
        @test out.attention_mask == flux["attention_mask"]
        dir = D.tokenizer_dir(Metalhead.subsource(Metalhead.model_source(repo, nothing),
            "tokenizer"))
        pipe = D.Flux2KleinPipeline(nothing, nothing, nothing, dir, nothing, true)
        @test D.tokenize(pipe, [prompt]) == (out.input_ids, out.attention_mask)
    end
end
