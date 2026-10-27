using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup
using Flux: Flux, f16

const ST = Metalhead.SentenceTransformers
const ST_FIXTURES = ["st_qwen3_lasttoken", "st_qwen3_dense", "st_qwen3_leftpad"]
metadata(file) = SafeTensors.deserialize(file).metadata

# The tokenizer of the fixtures, from porting/dump_sentence_transformers.py: the words "0",
# "1", ... of `vocab` are their ids, followed by special tokens, the first of which pads;
# the template adds `front` and `back` around a text.
struct WordsTokenizer
    vocab::Int
    front::Vector{Int}
    back::Vector{Int}
end
const SPECIALS = ["<|pad|>", "<|unk|>", "<|cls|>", "<|sep|>", "<|eos|>"]
function Metalhead.encode_ids(t::WordsTokenizer, text::AbstractString;
        add_special_tokens::Bool = true)
    ids = [something(tryparse(Int, w), -1) for w in split(text)]
    ids = [0 <= i < t.vocab ? i : t.vocab + 1 for i in ids]
    return add_special_tokens ? [t.front; ids; t.back] : ids
end
Metalhead.pad_token_id(t::WordsTokenizer) = t.vocab

function words_tokenizer(dir)
    tokenizer = JSON.parsefile(joinpath(dir, "tokenizer.json"))
    vocab = count(k -> tryparse(Int, k) !== nothing, keys(tokenizer["model"]["vocab"]))
    single = tokenizer["post_processor"]["single"]
    i = findfirst(x -> haskey(x, "Sequence"), single)
    id(x) = vocab + findfirst(==(x["SpecialToken"]["id"]), SPECIALS) - 1
    return WordsTokenizer(vocab, id.(single[1:(i - 1)]), id.(single[(i + 1):end]))
end

# `model` with the tokenizer `tok` in its Transformer module
function with_tokenizer(model, tok)
    t = model[1]
    transformer = ST.Transformer(t.auto_model, tok, t.max_seq_length, t.do_lower_case,
        t.truncation_side)
    return ST.SentenceTransformer(
        (transformer, Base.tail(model.modules)...), model.prompts,
        model.default_prompt_name, model.similarity_fn_name, model.truncate_dim)
end

fixture_spec(dir) = JSON.parse(metadata(joinpath(dir, "acts.safetensors"))["spec"])

# The prompt that the call `call` of a fixture's spec resolves, as `encode_query` and
# `encode_document` resolve it
function call_prompt(model, call)
    prompt, name = get(call, "prompt", nothing), get(call, "prompt_name", nothing)
    method = get(call, "method", "encode")
    if prompt === nothing && name === nothing
        candidates = method == "encode_query" ? (:query,) :
                     method == "encode_document" ? (:document, :passage, :corpus) : ()
        found = findfirst(k -> haskey(model.prompts, k), candidates)
        found === nothing || (name = candidates[found])
    end
    return ST.resolve_prompt(model, prompt, name)
end

const ENCODERS = Dict("encode" => ST.encode, "encode_query" => encode_query,
    "encode_document" => encode_document)

# The columns of `x`, of size (..., seq_len, batch), at the tokens of each text, those of
# its `mask`: sentence-transformers pads on the left where the tokenizer says so, and
# Metalhead on the right.
function tokens_of(x, mask)
    return [selectdim(selectdim(x, ndims(x), j), ndims(x) - 1, findall(==(1), mask[:, j]))
            for j in axes(mask, 2)]
end

# The relative errors of each call of the fixture `dir` of `model` (eltype Float64): of the
# outputs of the model on the inputs that `preprocess` makes of all the texts, and of its
# `encode` in batches, and whether those inputs are sentence-transformers', up to the side of
# the padding (of the tokens of the texts alone, where it is not the same).
function check_st_fixture(dir, model)
    acts = Metalhead.Checkpoint(joinpath(dir, "acts.safetensors"))
    spec = fixture_spec(dir)
    texts = String.(spec["texts"])
    left = get(spec["tokenizer"], "padding_side", "right") == "left"
    return map(enumerate(spec["calls"])) do (i, call)
        key(k) = "calls/$i/$k"
        kwargs = (; (Symbol(k) => v for (k, v) in call if k != "method")...)
        embeddings = ENCODERS[get(call, "method", "encode")](model, texts;
            batch_size = spec["batch_size"], kwargs...)
        inputs = ST.preprocess(model, texts; prompt = call_prompt(model, call))
        outputs = model(inputs)
        ids, mask = acts[key("input_ids")] .+ 1, acts[key("attention_mask")]
        expected = acts[key("token_embeddings")]
        same_inputs = (left ?
                       tokens_of(inputs.input_ids, inputs.attention_mask) ==
                       tokens_of(ids, mask) :
                       inputs.input_ids == ids && inputs.attention_mask == mask) &&
                      get(inputs, :prompt_length, nothing) ==
                      (haskey(acts, key("prompt_length")) ? acts[key("prompt_length")][] :
                       nothing)
        token_embeddings = if left
            relerr(
                reduce(hcat, tokens_of(outputs.token_embeddings, inputs.attention_mask)),
                reduce(hcat, tokens_of(expected, mask)))
        else
            relerr(outputs.token_embeddings, expected)
        end
        return (; same_inputs, token_embeddings,
            sentence_embedding = relerr(outputs.sentence_embedding,
                acts[key("sentence_embedding")]),
            embeddings = relerr(embeddings, acts[key("embeddings")]))
    end
end

function fixture_st(name; eltype = Float64)
    return with_tokenizer(create_model(fixture(name); eltype),
        words_tokenizer(fixture(name)))
end

@testset "fixtures" begin
    for name in ST_FIXTURES
        model = @test_logs fixture_st(name)
        @test model isa ST.SentenceTransformer
        for r in check_st_fixture(fixture(name), model)
            @test r.same_inputs
            @test r.token_embeddings <= 1e-8
            @test r.sentence_embedding <= 1e-8
            @test r.embeddings <= 1e-8
        end
    end
    # Qwen3-Embedding's: a Qwen3ForCausalLM checkpoint (with an untied head, which AutoModel
    # doesn't load), the max_seq_length of sentence-transformers 5, in tokenizer_config.json
    model = fixture_st("st_qwen3_lasttoken")
    @test model[1].auto_model isa Metalhead.Transformers.LlamaModel
    @test model[1].max_seq_length == 9 && model[1].truncation_side == "right"
    @test model[2].pooling_mode == ("lasttoken",) && model[3] isa ST.Normalize
    @test model.prompts == (; document = "", query = "31 32 Query:")
    @test model.default_prompt_name === nothing && model.similarity_fn_name == "cosine"
    # a tokenizer that pads on the left, pooling the last token and the weighted mean of the
    # tokens after the prompt, which the right padding of Metalhead gives the same embeddings of
    model = fixture_st("st_qwen3_leftpad")
    @test model[2].pooling_mode == ("lasttoken", "weightedmean") && !model[2].include_prompt
    # sentence-transformers < 5: legacy keys, the max_seq_length of
    # sentence_bert_config.json over the model_max_length of tokenizer_config.json, a
    # Normalize without config.json
    model = fixture_st("st_qwen3_dense")
    @test model[1].max_seq_length == 8 && model[1].do_lower_case
    @test model[2].pooling_mode == ("mean",) && !model[2].include_prompt
    @test [m.activation_function for m in model.modules[3:7]] ==
          [tanh, identity, Flux.relu, Flux.gelu_erf, Flux.sigmoid]
    @test model[4].residual === nothing && model[4].use_residual
    @test size(model[5].residual.weight) == (8, 12) && model[5].residual.bias === false
    @test model.default_prompt_name == "query" && model.truncate_dim == 4
    @test model.similarity_fn_name == "dot"
    # a single text gives a vector, and a batch of one the same embeddings
    texts = ["5 6 7", "8 9"]
    @test ST.encode(model, texts[1]) ≈ ST.encode(model, texts)[:, 1]
    @test ST.encode(model, texts; batch_size = 1) ≈ ST.encode(model, texts)
    @test size(ST.encode(model, String[])) == (0, 0)
    @test_throws "batch_size" ST.encode(model, texts; batch_size = 0)
    @test_throws "no prompt named \"other\"" ST.encode(model, texts; prompt_name = "other")
    @test_throws "a string or a vector of strings" ST.encode(model, 1)
    @test_throws "needs the inputs `input_ids`" model((; attention_mask = ones(Int, 2, 2)))
end

@testset "pooling" begin
    file = joinpath(fixture("st_pooling"), "acts.safetensors")
    acts = Metalhead.Checkpoint(file)
    meta = metadata(file)
    modes = String.(JSON.parse(meta["modes"]))
    prompt_length = JSON.parse(meta["spec"])["pooling"]["prompt_length"]
    x = acts["token_embeddings"]
    # padding on the right and on the left, with the prompt or not; a text of one token
    # without its prompt has no token, whose max is -Inf
    for side in ("right", "left"), mode in [modes; "all"], include_prompt in (true, false)
        pooling = ST.Pooling(size(x, 1); include_prompt,
            pooling_mode = mode == "all" ? modes : mode)
        y = pooling((; token_embeddings = x, attention_mask = acts["$side/attention_mask"],
            prompt_length)).sentence_embedding
        ref = acts["$side/$mode/$include_prompt"]
        @test isinf.(y) == isinf.(ref) && y[isinf.(y)] == ref[isinf.(ref)]
        @test relerr(y[isfinite.(y)], ref[isfinite.(ref)]) <= 1e-12
    end
    @test_throws "is not a pooling mode" ST.Pooling(4; pooling_mode = "first")
    # without a mask, every token is pooled
    y = ST.Pooling(5)((; token_embeddings = x)).sentence_embedding
    @test y ≈ dropdims(sum(x; dims = 2); dims = 2) ./ size(x, 2)
end

@testset "gradient" begin
    for name in ST_FIXTURES
        model = fixture_st(name; eltype = Float32)
        inputs = ST.preprocess(model, ["5 6 7", "8 9 10 11", "12"]; prompt = "1 2 ")
        @test check_gradient(model, inputs)
    end
end

@testset "half precision" begin
    for name in ST_FIXTURES
        model = fixture_st(name; eltype = Float32)
        inputs = ST.preprocess(model, ["5 6 7", "8 9 10 11", "12"]; prompt = "1 2 ")
        half = f16(model)
        @test eltype(half[1].auto_model.norm.weight) == Float32
        y = (@test_logs half(inputs)).sentence_embedding
        @test eltype(y) == Float16
        @test relerr(Float32.(y), model(inputs).sentence_embedding) <= 1e-2
    end
end

# A copy of the fixture `name` whose files `changes` (name => Dict, a string, or `nothing`
# to remove it) are replaced
function with_changes(f, name, changes...)
    mktempdir() do dir
        cp(fixture(name), dir; force = true)
        for (file, content) in changes
            path = joinpath(dir, file)
            if content === nothing
                rm(path)
            else
                write(path, content isa AbstractString ? content : JSON.json(content))
            end
        end
        return f(dir)
    end
end

read_fixture(name, file) = JSON.parsefile(joinpath(fixture(name), file))
changed(name, file, kv...) = file => merge(read_fixture(name, file), Dict(kv...))

@testset "configuration" begin
    using Metalhead: model_source, source_config, identify
    using .ST: SENTENCE_TRANSFORMERS
    dir = fixture("st_qwen3_lasttoken")
    @test string(SENTENCE_TRANSFORMERS) == "sentence-transformers"
    @test identify(source_config(model_source(dir, nothing))) ==
          (SENTENCE_TRANSFORMERS, "SentenceTransformer")
    @test "SentenceTransformer" in list_models(; byfamily = true)["sentence_transformers"]
    @test_throws "built from a repository with a modules.json" create_model("SentenceTransformer")
    @test_throws "has no processor" create_processor(dir)

    # overrides of the settings of config_sentence_transformers.json
    model = create_model(
        dir; truncate_dim = 3, prompts = Dict("query" => "1 ", "other" => "2 "),
        default_prompt_name = "other", similarity_fn_name = "euclidean")
    @test model.truncate_dim == 3 && model.similarity_fn_name == "euclidean"
    @test model.prompts == (; document = "", query = "1 ", other = "2 ")
    @test model.default_prompt_name == "other"
    @test_throws "`num_hidden_layers` is not a setting of a SentenceTransformer" create_model(
        dir; num_hidden_layers = 1)
    @test_throws "is not a prompt of the model" create_model(dir; default_prompt_name = "x")
    @test_throws "`truncate_dim` is a positive integer" create_model(dir; truncate_dim = 0)
    # sentence-transformers uses cosine for the similarities of other models
    @test create_model(dir; similarity_fn_name = "maxsim").similarity_fn_name == "cosine"
    @test_throws "has no tokenizer.json" ST.preprocess(
        with_tokenizer(model, nothing), ["1"])
    deferred = Metalhead.DeferredTokenizer(joinpath(dir, "tokenizer.json"), "<|pad|>")
    if Base.get_extension(Metalhead, :MetalheadHuggingFaceTokenizersExt) === nothing
        @test model[1].tokenizer == deferred
        @test_throws "run `using HuggingFaceTokenizers`" ST.encode(model, ["1"])
    end

    # random weights, and the arrays of each module in the eltype
    random = create_model(dir; pretrained = false)
    @test random[1].auto_model.norm.weight == ones(Float32, 16)
    @test eltype(create_model(fixture("st_qwen3_dense"); eltype = Float16)[3].linear.weight) ==
          Float16
    half = create_model(dir; eltype = Float16)
    @test eltype(half[1].auto_model.embed_tokens.weight) == Float16
    @test eltype(half[1].auto_model.norm.weight) == Float32

    # strictly read: unknown keys warn once, unsupported values are errors
    with_changes("st_qwen3_lasttoken",
        changed("st_qwen3_lasttoken", "1_Pooling/config.json", "pooling_spelling" => 1)) do d
        @test_logs (:warn, r"`pooling_spelling` in config.json is not a setting of Pooling") create_model(d)
    end
    unsupported = [
        changed("st_qwen3_lasttoken", "sentence_bert_config.json",
            "transformer_task" => "sequence-classification") => "`transformer_task = \"sequence-classification\"` in sentence_bert_config.json is not supported",
        changed("st_qwen3_lasttoken", "sentence_bert_config.json",
            "modality_config" => Dict("message" => Dict())) => "`modality_config",
        changed("st_qwen3_lasttoken", "sentence_bert_config.json",
            "processing_kwargs" => Dict("text" => Dict("max_length" => 4))) => "`processing_kwargs",
        changed("st_qwen3_lasttoken", "sentence_bert_config.json",
            "query_length" => 4) => "`query_length = 4`",
        changed("st_qwen3_lasttoken", "sentence_bert_config.json",
            "tokenizer_args" => Dict("model_max_length" => 4)) => "`processor_kwargs",
        changed("st_qwen3_lasttoken", "sentence_bert_config.json",
            "do_lower_case" => true) => "`do_lower_case = true`",
        changed("st_qwen3_lasttoken", "1_Pooling/config.json",
            "pooling_mode" => "first") => "`pooling_mode = \"first\"`",
        changed("st_qwen3_lasttoken", "config_sentence_transformers.json",
            "model_type" => "SparseEncoder") => "`model_type = \"SparseEncoder\"`",
        changed("st_qwen3_lasttoken", "tokenizer_config.json",
            "truncation_side" => "middle") => "`truncation_side = \"middle\"`",
        changed("st_qwen3_lasttoken", "config.json",
            "model_type" => "bert") => "`model_type = \"bert\"`, whose AutoModel isn't ported",
        changed("st_qwen3_dense", "2_Dense/config.json",
            "activation_function" => "torch.nn.modules.activation.Softplus") => "`activation_function = \"torch.nn.modules.activation.Softplus\"`"]
    for (change, message) in unsupported
        name = startswith(first(change), "2_Dense") ? "st_qwen3_dense" :
               "st_qwen3_lasttoken"
        with_changes(d -> (@test_throws message create_model(d)), name, change)
    end
    modules = read_fixture("st_qwen3_lasttoken", "modules.json")
    for (type, message) in ("sentence_transformers.models.WordWeights" => "has not been ported",
        "custom_module.Pooling" => "remote code")
        entries = [modules[1:1]; [merge(modules[2], Dict("type" => type))]; modules[3:end]]
        with_changes(d -> (@test_throws message create_model(d)), "st_qwen3_lasttoken",
            "modules.json" => entries)
    end
    with_changes(d -> (@test_throws "`kwargs" create_model(d)), "st_qwen3_lasttoken",
        "modules.json" => [merge(modules[1], Dict("kwargs" => ["task"])); modules[2:end]])
    # a tokenizer that pads on the left, as that of the fixture st_qwen3_leftpad, which a
    # causal model supports
    with_changes("st_qwen3_lasttoken",
        changed("st_qwen3_lasttoken", "tokenizer_config.json", "padding_side" => "left")) do d
        @test create_model(d) isa ST.SentenceTransformer
    end
    # the Dense weights are safetensors
    with_changes(d -> (@test_throws "loads safetensors weights only" create_model(d)),
        "st_qwen3_dense", "2_Dense/model.safetensors" => nothing)
    # the model_max_length of tokenizer_config.json, capped at max_position_embeddings
    with_changes("st_qwen3_lasttoken",
        changed("st_qwen3_lasttoken", "tokenizer_config.json", "model_max_length" => 1e30),
        changed("st_qwen3_lasttoken", "config.json", "max_position_embeddings" => 6)) do d
        @test create_model(d)[1].max_seq_length == 6
    end
end

@testset "truncation" begin
    tok = WordsTokenizer(40, [42], [43])
    @test ST.encode_truncated(tok, "1 2 3 4", 4, "right") == [42, 1, 2, 43]
    @test ST.encode_truncated(tok, "1 2 3 4", 4, "left") == [42, 3, 4, 43]
    @test ST.encode_truncated(tok, "1 2", 4, "right") == [42, 1, 2, 43]
    @test ST.encode_truncated(tok, "1 2 3 4", nothing, "right") == [42, 1, 2, 3, 4, 43]
    @test_throws "more than `max_seq_length = 1`" ST.encode_truncated(tok, "1 2", 1, "right")
    @test ST.prompt_length(fixture_st("st_qwen3_dense")[1], tok, "1 2 ") == 3
end

@testset "similarity" begin
    a, b = [3.0 0; 4 1], [1.0 0 2; 0 1 2]
    sim(name) = similarity(
        create_model(fixture("st_qwen3_lasttoken");
            similarity_fn_name = name), a, b)
    @test sim("cosine") ≈ [0.6 0.8 0.7 * sqrt(2); 0 1 sqrt(0.5)]
    @test sim("dot") == [3 4 14; 0 1 2]
    @test sim("euclidean") ≈ -[sqrt(20) sqrt(18) sqrt(5); sqrt(2) 0 sqrt(5)]
    @test sim("manhattan") == -[6 6 3; 2 0 3]
    @test size(sim("dot")[:, 1:1]) ==
          size(similarity(create_model(fixture(
                  "st_qwen3_lasttoken")), a, b[:, 1])) == (2, 1)
end

# HuggingFaceTokenizers.jl, which METALHEAD_TEST_NETWORK adds to the test environment
TEST_NETWORK && @eval using HuggingFaceTokenizers: HuggingFaceTokenizers

TEST_NETWORK && @testset "tokenizers and references" begin
    encode = HuggingFaceTokenizers.encode
    # the tokenizer.json of each fixture gives the inputs of sentence-transformers
    for name in ST_FIXTURES
        model = @test_logs create_model(fixture(name); eltype = Float64)
        @test model[1].tokenizer isa HuggingFaceTokenizers.Tokenizer
        for r in check_st_fixture(fixture(name), model)
            @test r.same_inputs
            @test r.embeddings <= 1e-8
        end
        texts = ["5 6 7", "8 9"]
        @test encode(model, texts) == ST.encode(model, texts)
    end
    # sentence-transformers' embeddings of the queries and documents of each reference, in
    # float32, in one batch
    for file in reference_files_of("SentenceTransformer")
        meta = metadata(file)
        ref = Metalhead.Checkpoint(file)
        model = @test_logs create_model(meta["repo"]; revision = meta["revision"])
        embeddings = Dict()
        for (method, texts) in JSON.parse(meta["calls"])
            texts = String.(texts)
            embeddings[method] = method == "encode" ? encode(model, texts) :
                                 ENCODERS[method](model, texts)
            @test relerr(embeddings[method], ref["$method/embeddings"]) <= 1e-5
            prompt = call_prompt(model, Dict("method" => method))
            inputs = ST.preprocess(model, texts; prompt)
            @test inputs.input_ids == ref["$method/input_ids"] .+ 1
            @test inputs.attention_mask == ref["$method/attention_mask"]
        end
        # the reference stores torch's (queries, documents) matrix, reversed
        s = similarity(model, embeddings["encode_query"], embeddings["encode_document"])
        @test relerr(s, permutedims(ref["similarity"])) <= 1e-5
    end
end
