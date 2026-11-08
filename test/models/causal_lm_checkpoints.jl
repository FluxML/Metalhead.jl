using Metalhead, Test
using SafeTensors: SafeTensors
include(joinpath(@__DIR__, "..", "data.jl"))
using .TestData: fixture

# Calls `f(dir)` on a copy of the fixture `name` whose checkpoint has the keys `rename` gives,
# without those for which `drop` is true, and the tensors of `add`.
function with_checkpoint(f, name; rename = identity, drop = Returns(false),
        add = Dict{String, AbstractArray}())
    mktempdir() do dir
        src = fixture(name)
        cp(joinpath(src, "config.json"), joinpath(dir, "config.json"))
        st = SafeTensors.deserialize(joinpath(src, "model.safetensors"))
        tensors = Dict{String, AbstractArray}(
            rename(k) => copy(st[k]) for k in keys(st) if !drop(k))
        merge!(tensors, add)
        SafeTensors.serialize(joinpath(dir, "model.safetensors"), tensors)
        return f(dir)
    end
end

function fixture_keys(name)
    return collect(keys(SafeTensors.deserialize(
        joinpath(fixture(name), "model.safetensors"))))
end

# The tokens of a batch for the model `m`
tokens(m) = rand(1:size(Metalhead.Transformers.decoder(m)[1].embed_tokens.weight, 2), 5, 2)

@testset "Llama-2-era checkpoints store rotary_emb.inv_freq" begin
    for name in ("llama_tiny", "qwen2_tiny", "qwen3_tiny", "qwen3_moe_tiny", "gemma3_tiny",
        "granite_tiny", "granitemoe_tiny", "granitemoeshared_tiny", "granitemoehybrid_tiny")
        layers = unique(only(match(r"^model\.layers\.(\d+)\.", k).captures)
        for k in fixture_keys(name) if startswith(k, "model.layers."))
        inv_freq = Dict{String, AbstractArray}(
            "model.layers.$i.self_attn.rotary_emb.inv_freq" => Float32[1, 0.1]
        for i in layers)
        ref = create_model(fixture(name); eltype = Float64)
        ids = tokens(ref)
        with_checkpoint(name; add = inv_freq) do dir
            @test create_model(dir; eltype = Float64)(ids) == ref(ids)
        end
        # only that buffer is dropped
        other = Dict{String, AbstractArray}(
            "model.layers.0.self_attn.rotary_emb.cos_cached" => Float32[1, 0.1])
        with_checkpoint(name; add = other) do dir
            @test_throws "unused:     model.layers.0.self_attn.rotary_emb.cos_cached" create_model(dir)
        end
    end
end

# transformers' `base_model_prefix`: a checkpoint saved from the base model (Qwen3Model) of a
# `...ForCausalLM` config has the keys of `model.*` without the prefix and no `lm_head`.
base_key(k) = startswith(k, "model.") ? k[7:end] : k
is_head(k) = startswith(k, "lm_head.")

@testset "base-model checkpoints under a causal LM config" begin
    tied = ("llama_tiny", "qwen2_tiny", "qwen3_tiny", "gemma3_tiny", "granite_tiny",
        "granitemoe_tiny", "granitemoeshared_tiny", "granitemoehybrid_tiny",
        "qwen3_5_text_tiny")
    untied = ("llama_options", "qwen3_moe_tiny", "qwen3_5_moe_text_tiny", "qwen3_5_tiny")
    for name in (tied..., untied...)
        @testset "$name" begin
            ref = create_model(fixture(name); eltype = Float64)
            ids = tokens(ref)
            with_checkpoint(name; rename = base_key, drop = is_head) do dir
                if name in tied
                    # the tied head is the embedding matrix itself
                    model = @test_logs create_model(dir; eltype = Float64)
                    @test model.lm_head.weight === model.model.embed_tokens.weight
                    @test model(ids) == ref(ids)
                else
                    model = @test_logs (:warn, r"`lm_head.weight` is newly initialized") create_model(
                        dir; eltype = Float64)
                    @test size(model.lm_head.weight) == size(ref.lm_head.weight)
                    @test all(isfinite, model.lm_head.weight)
                    @test model.lm_head.weight != ref.lm_head.weight
                end
                @test Metalhead.forward_features(model, ids) ==
                      Metalhead.forward_features(ref, ids)
            end
        end
    end

    @testset "the head, if the checkpoint has one" begin
        ref = create_model(fixture("llama_options"); eltype = Float64)
        ids = tokens(ref)
        with_checkpoint("llama_options"; rename = base_key) do dir
            model = @test_logs create_model(dir; eltype = Float64)
            @test model(ids) == ref(ids)
        end
    end

    @testset "half precision" begin
        with_checkpoint("llama_options"; rename = base_key, drop = is_head) do dir
            model = @test_logs (:warn, r"newly initialized") create_model(
                dir; eltype = Float16)
            @test eltype(model.lm_head.weight) == Float16
            @test all(isfinite, model.lm_head.weight)
            @test eltype(model.model.embed_tokens.weight) == Float16
        end
    end

    @testset "other checkpoints still fail" begin
        name = "llama_options"
        # a checkpoint of the model itself without its untied head
        with_checkpoint(name; drop = is_head) do dir
            @test_throws "missing:    lm_head.weight" create_model(dir)
        end
        # a base-model checkpoint with a weight missing, or with an unknown one
        with_checkpoint(name; rename = base_key,
            drop = k -> is_head(k) || k == "model.layers.0.mlp.up_proj.weight") do dir
            @test_throws "missing:    layers.0.mlp.up_proj.weight" create_model(dir)
        end
        extra = Dict{String, AbstractArray}("layers.0.bogus" => Float32[1])
        with_checkpoint(name; rename = base_key, add = extra) do dir
            @test_throws "unused:     layers.0.bogus" create_model(dir)
        end
        # a checkpoint of another layout
        with_checkpoint(name; rename = k -> "bert." * k) do dir
            @test_throws "missing:    model.embed_tokens.weight" create_model(dir)
        end
        # prefixed keys are not mixed with unprefixed ones
        with_checkpoint(name; add = Dict{String, AbstractArray}("norm.weight" =>
            Float32[1])) do dir
            @test_throws "unused:     norm.weight" create_model(dir)
        end
    end
end
