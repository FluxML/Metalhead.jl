using Metalhead, Test, Random
include("../parity_setup.jl")
using .ParitySetup
using Metalhead: Checkpoint
using Metalhead.Transformers: GenerationConfig, process_logits, excludes_eos, sample_tokens

# transformers' `generate` on random prompts (porting/dump_generation.py), with a key-value
# cache (Qwen3) and without (Gemma 3)
@testset "$name" for name in ["generate_qwen3", "generate_gemma3"]
    dir = fixture(name)
    model = fixture_model(dir)
    file = joinpath(dir, "acts.safetensors")
    acts, meta = Checkpoint(file), SafeTensors.deserialize(file).metadata
    spec = JSON.parse(meta["spec"])["generate"]
    settings = (; (Symbol(k) => v for (k, v) in spec["generation_config"])...)
    eos = Tuple(JSON.parse(meta["eos_token_id"]))
    @test model.generation_config == GenerationConfig(; settings..., eos_token_id = eos)
    ids, mask = acts["input_ids"] .+ 1, acts["attention_mask"]
    lengths = vec(sum(mask; dims = 1))
    batch_size = length(lengths)

    @testset "$(JSON.json(case))" for (i, case) in enumerate(spec["cases"])
        kwargs = (; (Symbol(k) => v for (k, v) in case)...)
        c = GenerationConfig(model.generation_config; kwargs...)
        expected = [acts["cases/$i/$b/tokens"] .+ 1 for b in 1:batch_size]
        if c.do_sample
            # the logits processors, on transformers' logits of each step
            for b in 1:batch_size
                logits, scores = acts["cases/$i/$b/logits"], acts["cases/$i/$b/scores"]
                vocab = axes(logits, 1)
                is_eos = vocab .∈ Ref(collect(c.eos_token_id) .+ 1)
                for t in axes(logits, 2)
                    tokens = [ids[1:lengths[b], b]; expected[b][1:(t - 1)]]
                    seen = reshape(vocab .∈ Ref(tokens), :, 1)
                    got = process_logits(c, logits[:, t:t], seen, is_eos,
                        excludes_eos(c, lengths[b], t))
                    ref = scores[:, t:t]
                    @test isinf.(got) == isinf.(ref)
                    @test got[isfinite.(got)] ≈ ref[isfinite.(ref)] rtol = 1e-6
                end
            end
        else
            # the outputs of each sequence, padded after its end in a batch
            n = maximum(length, expected)
            pad = something(c.pad_token_id, first(c.eos_token_id)) + 1
            @test generate(model, ids; attention_mask = mask, kwargs...) ==
                  [t <= length(e) ? e[t] : pad for t in 1:n, e in expected]
            for b in 1:batch_size
                @test generate(model, ids[1:lengths[b], b]; kwargs...) == expected[b]
            end
        end
    end

    # a processor's output, and a vector that the call leaves resizable (Julia 1.10)
    @test generate(model, (; input_ids = ids, attention_mask = mask)) ==
          generate(model, ids; attention_mask = mask)
    sequence = ids[1:lengths[1], 1]
    @test generate(model, (; input_ids = sequence, attention_mask = ones(Int, lengths[1]))) ==
          generate(model, sequence)
    @test length(push!(sequence, 1)) == lengths[1] + 1

    # past the 256 tokens after the prompt that a cache starts with, which then grows
    long = (; max_new_tokens = 300, eos_token_id = nothing, repetition_penalty = 1.0)
    @test [sequence[1:lengths[1]]; generate(model, sequence[1:lengths[1]]; long...)] ==
          Parity.greedy_decode(model, sequence[1:lengths[1]], 300)

    # sampling: reproducible, and greedy when only the most likely token is kept
    sample(seed; kwargs...) = generate(model, ids; attention_mask = mask, do_sample = true,
        rng = Xoshiro(seed), kwargs...)
    @test sample(1) == sample(1)
    @test sample(1; top_k = 1) == generate(model, ids; attention_mask = mask)
end

@testset "sampler" begin
    p = [0.1, 0.2, 0.3, 0.4, 0.0]
    tokens = sample_tokens(Xoshiro(0), repeat(log.(Float32.(p)), 1, 40_000))
    @test [count(==(i), tokens) / length(tokens) for i in eachindex(p)] ≈ p atol = 0.01
end

@testset "settings" begin
    c = GenerationConfig(; do_sample = true, top_k = 20, eos_token_id = 2)
    @test c.eos_token_id == (2,) && c.temperature == 1 && c.max_new_tokens === nothing
    @test sprint(show, c) == "GenerationConfig(do_sample = true, top_k = 20, eos_token_id = (2,))"
    @test GenerationConfig(c; eos_token_id = [3, 4], top_p = 0.9) ==
          GenerationConfig(; do_sample = true, top_k = 20, top_p = 0.9, eos_token_id = (3, 4))

    # without generation_config.json, the token ids of config.json; with it, strictly
    @test fixture_model(fixture("llama_tiny")).generation_config ==
          GenerationConfig(; bos_token_id = 1, eos_token_id = 2)
    @test create_model(fixture("generate_qwen3"); pretrained = false).generation_config ==
          GenerationConfig()
    dir = mktempdir()
    for f in ("config.json", "model.safetensors")
        cp(joinpath(fixture("generate_qwen3"), f), joinpath(dir, f))
    end
    write_settings(d) = write(joinpath(dir, "generation_config.json"), JSON.json(d))
    write_settings(Dict("num_beams" => 1, "renormalize_logits" => true, "top_k" => nothing,
        "eos_token_id" => [3, 4], "transformers_version" => "5.0"))
    @test create_model(dir).generation_config == GenerationConfig(; eos_token_id = (3, 4))
    write_settings(Dict("num_beams" => 4))
    @test_throws "`num_beams = 4` in generation_config.json is not supported" create_model(dir)
    write_settings(Dict("chat_format" => "x"))
    @test_logs (:warn, r"`chat_format` in generation_config.json") create_model(dir)

    model = fixture_model(fixture("llama_tiny"))
    ids = [3, 4, 5]
    @test_throws "`num_beams` is not supported" generate(model, ids; num_beams = 2)
    @test_throws "not a setting of `generate`" generate(model, ids; pixel_values = 1)
    @test_throws "`temperature` must be positive" generate(model, ids; do_sample = true,
        temperature = 0)
    @test_throws "`max_new_tokens` must be positive" generate(model, ids; max_new_tokens = 0)
    @test_throws "must pad on the right" generate(model, [1 2; 3 4];
        attention_mask = [0 1; 1 1])
    # 20 new tokens by default, `max_length` counting the prompt
    @test length(generate(model, ids; eos_token_id = nothing)) == 20
    @test length(generate(model, ids; eos_token_id = nothing, max_length = 8)) == 5
end
