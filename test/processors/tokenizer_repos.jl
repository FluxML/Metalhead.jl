using Metalhead, Test
using JSON: JSON
using SafeTensors: SafeTensors
using Metalhead.Transformers: TokenizersBackend, decode
const T = Metalhead.Transformers

include(joinpath(@__DIR__, "..", "processors_setup.jl"))

# Written by `porting/dump_processor.py tokenizers`: `AutoTokenizer.from_pretrained(repo)`
# of the validated language models, of the FLUX.2 [klein] pipeline, of Qwen3.5, of SAM 3
# and of GPT-2, on TEXTS; and Qwen3.5's `AutoProcessor` with keywords of calls in its
# tokenizer_config.json
const PROCESSORS = joinpath(pkgdir(Metalhead), "test", "fixtures", "processors")
const FILE = joinpath(pkgdir(Metalhead), "test", "references", "processors",
    "tokenizers.safetensors")
const META = SafeTensors.deserialize(FILE).metadata
const REF = Metalhead.Checkpoint(FILE)
const TEXTS = JSON.parse(META["texts"])
const INFO = Dict(key => JSON.parse(META[key]) for key in JSON.parse(META["keys"]))
const KEYS = sort!(collect(keys(INFO)))
const PROCESSOR_CASES = sort!([k for k in keys(META) if startswith(k, "Qwen3VLProcessor/")])

ids(key, i; special = true) = REF["$key/$(special ? "ids" : "content_ids")/$i"]
calls(key) = sort!([split(k, "/")[2] for k in keys(META)
                    if startswith(k, "$key/") && count(==('/'), k) == 1])
python_bool(b) = b ? "True" : "False"
call_kwargs(r) = (; (Symbol(k) => v isa String ? Symbol(v) : v
                     for (k, v) in r if !(k in ("text", "create", "error")))...)
tokenizer_of(p) = p isa TokenizersBackend ? p : p.tokenizer

# A backend that knows the texts of the reference, and the special tokens of its tokenizer:
# a tokenizer backend is any type with methods of `encode_ids` and `decode_ids`. Skipping
# special tokens, `decode` first drops the named special tokens, which transformers marks
# special in the backend: the reference has tokenizer.json's text of the remaining ids.
struct ReferenceBackend
    key::String
end
function T.encode_ids(b::ReferenceBackend, text::AbstractString; add_special_tokens = true)
    i = findfirst(==(text), TEXTS)
    i === nothing && return [INFO[b.key]["special_ids"][text]]
    return ids(b.key, i; special = add_special_tokens)
end
function T.decode_ids(b::ReferenceBackend, token_ids::AbstractVector{<:Integer};
        skip_special_tokens = false)
    named = Set(values(INFO[b.key]["special_ids"]))
    for i in eachindex(TEXTS)
        decoded = INFO[b.key]["decoded"][string(i)][python_bool(skip_special_tokens)]
        ids(b.key, i) == token_ids && return decoded["raw"]
        skip_special_tokens && filter(!in(named), ids(b.key, i)) == token_ids &&
            return decoded["raw_unnamed"]
    end
    return error("no text of the reference has the ids $token_ids")
end

# The fixture files of a tokenizer, with a tokenizer.json that has only the type of its
# model, which decides the clean-up of decoded text.
function with_fixture(f, key)
    m = INFO[key]
    mktempdir() do dir
        cp(joinpath(PROCESSORS, replace(m["repo"], "/" => "--")), dir; force = true)
        write(joinpath(dir, m["subfolder"], "tokenizer.json"),
            JSON.json(Dict("model" => Dict("type" => m["model_type"]))))
        return f(dir, m["subfolder"])
    end
end

# The checks of the tokenizer `key` made by `make(; overrides...)`: the settings, the ids of
# each text, padding, truncation, the overrides of the calls, and decoding. Returns the
# outputs, for the documented keys.
function check_tokenizer(key, make)
    m = INFO[key]
    tok = make()
    outputs = []
    @testset "$key" begin
        for (k, v) in m["settings"]
            @test getfield(tok, Symbol(k)) == v
        end
        @test Dict(string(k) => v for (k, v) in pairs(tok.model_specific_special_tokens)) ==
              m["named_tokens"]
        @test tok.tokenizer_class == m["cls"]
        for i in eachindex(TEXTS), special in (true, false)
            @test tok(TEXTS[i]; add_special_tokens = special).input_ids ==
                  ids(key, i; special) .+ 1
        end
        for name in calls(key)
            r = JSON.parse(META["$key/$name"])
            create = (; (Symbol(k) => v for (k, v) in r["create"])...)
            t = isempty(create) ? tok : make(; create...)
            texts = TEXTS[r["text"]]
            @testset "$name" begin
                if haskey(r, "error")
                    @test occursin("padding token", r["error"])
                    @test_throws "no padding token" t(texts; call_kwargs(r)...)
                else
                    out = t(texts; call_kwargs(r)...)
                    @test out.input_ids == REF["$key/$name/input_ids"] .+ 1
                    @test out.attention_mask == REF["$key/$name/attention_mask"]
                    push!(outputs, out)
                end
            end
        end
        for i in eachindex(TEXTS), skip in (false, true)
            @test decode(tok, ids(key, i) .+ 1; skip_special_tokens = skip) ==
                  m["decoded"][string(i)][python_bool(skip)]["text"]
        end
        push!(outputs, tok(TEXTS[1]))
    end
    return outputs
end

@testset "fixtures" begin
    outputs = []
    for key in KEYS
        with_fixture(key) do dir, subfolder
            function make(; kwargs...)
                p = create_processor(dir; subfolder, tokenizer = ReferenceBackend(key),
                    kwargs...)
                return tokenizer_of(p)
            end
            @test_logs make()
            append!(outputs, check_tokenizer(key, make))
        end
    end
    check_documented_keys(TokenizersBackend, outputs)
end

@testset "special tokens" begin
    key = "unsloth--gemma-3-270m"
    with_fixture(key) do dir, _
        tok = tokenizer_of(create_processor(dir; tokenizer = ReferenceBackend(key)))
        # the post-processor of tokenizer.json adds BOS: transformers drops `add_bos_token`
        # of tokenizer_config.json
        @test tok.add_bos_token === tok.add_eos_token === nothing
        @test first(tok(TEXTS[1]).input_ids) == 2 + 1
        # as in transformers, setting one replaces the post-processor: no BOS
        eos = create_processor(dir; tokenizer = ReferenceBackend(key), add_eos_token = true)
        @test eos(TEXTS[1]).input_ids == [ids(key, 1; special = false); 1] .+ 1
        text = sprint(show, MIME("text/plain"), eos)
        @test occursin(r"add_eos_token = true +override \(class default: nothing\)", text)
        # token ids need no backend, and are padded with the ids of the special tokens
        ids_only = tokenizer_of(create_processor(dir))
        @test ids_only([[2, 5], [2]]).input_ids == [3 1; 6 3]
        @test_throws "HuggingFaceTokenizers" ids_only("text")
        # GemmaTokenizer's `mask_token`, which its files don't set, is special: decoding
        # skips it, as transformers does
        @test tok.mask_token == "<mask>"
        @test occursin(r"mask_token = \"<mask>\" +class default \(GemmaTokenizer\)",
            sprint(show, MIME("text/plain"), tok))
        i = findfirst(==("<mask> token <mask>"), TEXTS)
        @test decode(tok, ids(key, i) .+ 1; skip_special_tokens = true) == " token "
        # indices of any vector of integers
        @test decode(tok, Any[ids(key, 1)...] .+ 1) ==
              INFO[key]["decoded"]["1"]["False"]["text"]
    end
    # GPT-2's files don't set its special tokens, which GPT2Tokenizer does
    key = "openai-community--gpt2"
    with_fixture(key) do dir, _
        tok = @test_logs create_processor(dir; tokenizer = ReferenceBackend(key))
        @test tok.tokenizer_class == "GPT2Tokenizer"
        @test tok.bos_token == tok.eos_token == tok.unk_token == "<|endoftext|>"
        bos = create_processor(dir; tokenizer = ReferenceBackend(key), add_bos_token = true)
        @test bos(TEXTS[1]).input_ids == [50256; ids(key, 1)] .+ 1
        # the padding token of token ids, from tokenizer.json without a backend
        @test_throws "`<|endoftext|>` is not a token of GPT2Tokenizer: neither an added token (`added_tokens_decoder`) nor a token of its tokenizer.json" create_processor(dir;
            pad_token = "<|endoftext|>")([[1, 2, 3], [4]])
    end
    with_fixture(key) do dir, _
        json = Dict("model" => Dict("vocab" => Dict("hello" => 7), "merges" => []),
            "added_tokens" => [Dict("id" => 50256, "content" => "<|endoftext|>")])
        write(joinpath(dir, "tokenizer.json"), JSON.json(json))
        padded = create_processor(dir; pad_token = "<|endoftext|>")([[1, 2, 3], [4]])
        @test padded.input_ids == [2 5; 3 50257; 4 50257]
        @test create_processor(dir; pad_token = "hello")([[1, 2], [4]]).input_ids ==
              [2 5; 3 8]
        @test T.tokenizer_json(joinpath(dir, "tokenizer.json")).model_type == "BPE"
    end
end

# The keywords of calls in tokenizer_config.json are the defaults of the calls of a
# processor that combines the tokenizer (transformers' `_merge_kwargs`), not of the
# tokenizer's own: Qwen3.5's files with those of each case
function with_processor_case(f, name; tokenizer_json = false)
    m = JSON.parse(META[name])
    mktempdir() do dir
        qwen = joinpath(PROCESSORS, "Qwen--Qwen3.5-0.8B")
        foreach(file -> cp(joinpath(qwen, file), joinpath(dir, file)), readdir(qwen))
        config = JSON.parsefile(joinpath(dir, "tokenizer_config.json"))
        write(joinpath(dir, "tokenizer_config.json"), JSON.json(merge(config, m["init"])))
        tokenizer_json && cp(Metalhead.hf_download("Qwen/Qwen3.5-0.8B", "tokenizer.json";
                revision = INFO["Qwen--Qwen3.5-0.8B"]["revision"]),
            joinpath(dir, "tokenizer.json"); follow_symlinks = true)
        return f(dir, m)
    end
end

function check_processor_case(name, proc)
    m = JSON.parse(META[name])
    @test proc isa T.Qwen3VLProcessor && m["cls"] == "Qwen3VLProcessor"
    @test Dict(pairs(proc.tokenizer.init_kwargs)) ==
          Dict(Symbol(k) => v for (k, v) in m["init"])
    out = proc(; text = TEXTS[m["text"]], call_kwargs(m["call"])...)
    @test out.input_ids == REF["$name/input_ids"] .+ 1
    @test out.attention_mask == REF["$name/attention_mask"]
    @test out.mm_token_type_ids == REF["$name/mm_token_type_ids"]
    return out
end

@testset "processor call defaults" begin
    key = "Qwen--Qwen3.5-0.8B"
    @test length(PROCESSOR_CASES) == 4
    for name in PROCESSOR_CASES
        with_processor_case(name) do dir, m
            proc = @test_logs create_processor(dir; tokenizer = ReferenceBackend(key))
            @testset "$name" begin
                check_processor_case(name, proc)
            end
            # the tokenizer alone ignores them
            @test proc.tokenizer(TEXTS[3]).input_ids == ids(key, 3) .+ 1
        end
    end
    with_processor_case("Qwen3VLProcessor/max_length") do dir, _
        proc = create_processor(dir)
        @test_throws "longer than `max_length = 5`" proc(; text = ids(key, 3))
        @test proc(; text = ids(key, 3), max_length = nothing).input_ids == ids(key, 3) .+ 1
        @test_throws "Qwen3VLProcessor has no keyword `stride`" proc(; text = [1], stride = 2)
        @test_throws "`padding = 3`" create_processor(dir; init_kwargs = (; padding = 3))(;
            text = [1])
        @test_throws "`init_kwargs` has `foo`" create_processor(dir;
            init_kwargs = (; foo = 1))
        @test_throws "`pad_to_multiple_of = 8` in the `init_kwargs` of the tokenizer of Qwen3VLProcessor" create_processor(dir;
            init_kwargs = (; pad_to_multiple_of = 8))
        @test create_processor(dir; init_kwargs = (; stride = 2)) isa T.Qwen3VLProcessor
    end
end

# Strings through HuggingFaceTokenizers.jl, which METALHEAD_TEST_NETWORK adds to the test
# environment (runtests.jl), on each repository's own files at its validated revision
const TEST_NETWORK = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))
if TEST_NETWORK
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers
    @testset "processor call defaults with HuggingFaceTokenizers" begin
        for name in PROCESSOR_CASES
            with_processor_case(name; tokenizer_json = true) do dir, _
                @testset "$name" begin
                    check_processor_case(name, @test_logs create_processor(dir))
                end
            end
        end
    end
    # the tokenizer of a repository; that of Qwen3.5 alone, without its Qwen3VLProcessor
    function repo_tokenizer(m; kwargs...)
        if m["repo"] == "Qwen/Qwen3.5-0.8B"
            source = Metalhead.model_source(m["repo"], m["revision"], m["subfolder"])
            return Metalhead.load_processor(TokenizersBackend,
                Metalhead.ProcessorFiles(source); kwargs...)
        end
        return tokenizer_of(create_processor(m["repo"]; revision = m["revision"],
            subfolder = m["subfolder"], kwargs...))
    end
    @testset "HuggingFaceTokenizers" begin
        for key in KEYS
            m = INFO[key]
            make(; kwargs...) = repo_tokenizer(m; kwargs...)
            tok = @test_logs make()
            @test tok.tokenizer === nothing
            check_tokenizer(key, make)
        end
    end
end
