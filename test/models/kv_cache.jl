using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup
using Metalhead.Transformers: KVCache, last_tokens

# The language models whose layers are all full attention
const CACHE_FIXTURES = ["llama_tiny", "llama_options", "llama_rope", "qwen2_tiny",
    "qwen2_options", "qwen3_tiny", "qwen3_options", "qwen3_padded", "qwen3_moe_tiny",
    "qwen3_moe_options", "granite_tiny", "granite_options", "granitemoe_tiny",
    "granitemoe_options", "granitemoeshared_tiny", "granitemoehybrid_tiny",
    "granitemoehybrid_options"]

# The tolerance of the fixtures against PyTorch, for float64 models
const RTOL = 1e-8

# deterministic token indices of size (len, batch_size)
tokens(vocab, len, batch_size) = mod1.((1:len) .* 7 .+ (1:batch_size)' .* 13, vocab)

# the logits of `model` for a sequence alone, at the positions of the tokens
alone(model, ids) = model(ids)

@testset "$name" for name in CACHE_FIXTURES
    model = fixture_model(fixture(name))
    vocab = size(model.model.embed_tokens.weight, 2)
    ids = tokens(vocab, 12, 3)
    full = model(ids)

    @testset "prefill and steps of one token" begin
        cache = KVCache(model, 3, 16)
        @test cache.lengths == [0, 0, 0]
        logits, returned = model(ids[1:7, :]; cache)
        @test returned === cache && cache.lengths == [7, 7, 7]
        @test relerr(logits, full[:, 1:7, :]) <= RTOL
        for j in 8:12
            logits, _ = model(ids[j:j, :]; cache)
            @test relerr(logits, full[:, j:j, :]) <= RTOL
        end
        @test cache.lengths == [12, 12, 12]
    end

    @testset "steps of several tokens" begin
        cache = KVCache(model, 3, 12)
        for chunk in (1:3, 4:7, 8:8, 9:12)
            logits, _ = model(ids[chunk, :]; cache)
            @test relerr(logits, full[:, chunk, :]) <= RTOL
        end
    end

    @testset "sequences of different lengths" begin
        lengths = [12, 8, 5]
        mask = [i <= l for i in 1:12, l in lengths]
        cache = KVCache(model, 3, 20)
        logits, _ = model(ids; attention_mask = mask, cache)
        @test cache.lengths == lengths
        # the next-token logits are at the last token of each sequence, not at the end
        last = last_tokens(logits, cache.lengths)
        @test size(last) == (vocab, 3)
        steps = [tokens(vocab, 1, 3), tokens(vocab, 3, 3) .+ 1, tokens(vocab, 1, 3) .+ 2]
        steps = [mod1.(s, vocab) for s in steps]
        outputs = [logits]
        for step in steps
            push!(outputs, first(model(step; cache)))
        end
        @test cache.lengths == lengths .+ 5
        for b in 1:3
            # each sequence as it runs alone
            sequence = [ids[1:lengths[b], b]; reduce(vcat, [s[:, b] for s in steps])]
            ref = alone(model, sequence)
            @test relerr(logits[:, 1:lengths[b], b], ref[:, 1:lengths[b]]) <= RTOL
            @test relerr(last[:, b], ref[:, lengths[b]]) <= RTOL
            start = lengths[b]
            for (step, out) in zip(steps, outputs[2:end])
                n = size(step, 1)
                @test relerr(out[:, :, b], ref[:, (start + 1):(start + n)]) <= RTOL
                start += n
            end
        end
    end

    @testset "greedy loop" begin
        lengths = [9, 6, 4]
        mask = [i <= l for i in 1:9, l in lengths]
        prompts = ids[1:9, :]
        cache = KVCache(model, 3, 20)
        logits, _ = model(prompts; attention_mask = mask, cache)
        next = argmax.(eachcol(last_tokens(logits, cache.lengths)))
        generated = [next]
        for _ in 1:6
            logits, _ = model(reshape(next, 1, :); cache)
            next = argmax.(eachcol(dropdims(logits; dims = 2)))
            push!(generated, next)
        end
        for b in 1:3
            sequence = prompts[1:lengths[b], b]
            for _ in 1:7
                # not push!: on Julia 1.10 the model's reshape makes `sequence` unresizable
                sequence = [sequence; argmax(model(sequence)[:, end])]
            end
            @test sequence[(lengths[b] + 1):end] == getindex.(generated, b)
        end
    end
end

@testset "inputs and outputs" begin
    model = fixture_model(fixture("llama_tiny"))
    vocab = size(model.model.embed_tokens.weight, 2)
    ids = tokens(vocab, 6, 2)
    mask = [i <= l for i in 1:6, l in (6, 4)]

    # a vector is a sequence, and the output of a processor a NamedTuple
    cache = KVCache(model, 1, 8)
    logits, _ = model(ids[:, 1]; cache)
    @test size(logits) == (vocab, 6) && relerr(logits, model(ids[:, 1])) <= RTOL
    logits, _ = model(ids[1:1, 1]; cache)
    @test size(logits) == (vocab, 1) && cache.lengths == [7]
    cache = KVCache(model, 2, 8)
    logits, _ = model((; input_ids = ids, attention_mask = mask); cache)
    @test relerr(logits, model(ids; attention_mask = mask)) <= RTOL
    features, _ = forward_features(model, ids[1:1, :]; cache)
    @test size(features) == (size(model.model.norm.weight, 1), 1, 2)
    @test cache.lengths == [7, 5]

    # the features and logits agree
    cache = KVCache(model, 2, 8)
    features, _ = forward_features(model, ids; cache)
    @test relerr(features, forward_features(model, ids)) <= RTOL

    # the head on the last tokens only, as `last_tokens` of the logits
    @test relerr(forward_head(model, last_tokens(features, [6, 4])),
        last_tokens(model(ids; attention_mask = mask), [6, 4])) <= RTOL

    # an empty cache starts again
    empty!(cache)
    @test cache.lengths == [0, 0]
    logits, _ = model(ids; attention_mask = mask, cache)
    @test cache.lengths == [6, 4]
    @test relerr(logits, model(ids; attention_mask = mask)) <= RTOL

    # the buffers have the type of the weights
    @test eltype(first(cache.keys)) == Float64
    half = fixture_model(fixture("llama_tiny"); eltype = Float16)
    cache = KVCache(half, 2, 8)
    @test eltype(first(cache.values)) == Float16
    logits, _ = half(ids; cache)
    @test eltype(logits) == Float16 && all(isfinite, logits)
    @test sprint(show, cache) ==
          "KVCache(2 layers, 2 sequences of up to 8 tokens, lengths = [6, 6])"
end

@testset "errors" begin
    model = fixture_model(fixture("llama_tiny"))
    vocab = size(model.model.embed_tokens.weight, 2)
    ids = tokens(vocab, 6, 2)
    mask = [i <= l for i in 1:6, l in (6, 4)]
    cache = KVCache(model, 2, 8)
    model(ids; cache)
    @test_throws "don't fit" model(ids; cache)
    @test cache.lengths == [6, 6]
    @test_throws "only goes with the first call of an empty cache" model(ids[1:1, :];
        attention_mask = mask[1:1, :], cache)
    @test_throws "for 2 sequences" model(ids[:, 1:1]; cache)
    @test_throws "must pad on the right" model(ids; attention_mask = .!mask, cache = KVCache(
        model, 2, 8))
    @test_throws "must be positive" KVCache(model, 0, 8)
    @test_throws "must be positive" KVCache(model, 2, 0)
    @test_throws "the cache has 3 layers, the model 2" model.model(ids;
        cache = KVCache(fixture_model(fixture("llama_options")), 2, 8))

    # sliding-window attention and the gated delta rule have no cache yet
    for name in ["gemma3_tiny", "qwen3_5_text_tiny", "qwen3_5_tiny"]
        unsupported = fixture_model(fixture(name))
        @test_throws "not supported yet" KVCache(unsupported, 2, 8)
        @test_throws "not supported yet" unsupported(ids; cache)
    end
end
