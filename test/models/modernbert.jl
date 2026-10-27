using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup
using Flux: Flux
using Metalhead.Transformers: ModernBertModel, ModernBertForMaskedLM,
                              ModernBertForSequenceClassification, BiaslessLayerNorm

const MODERNBERT_FIXTURES = [
    "modernbert_tiny", "modernbert_masked_lm", "modernbert_options",
    "modernbert_classifier"]

fixture_acts(name) = Metalhead.Checkpoint(joinpath(fixture(name), "acts.safetensors"))

@testset "fixtures" begin
    for name in MODERNBERT_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
        # the parameters of transformers' model, all loaded by name
        meta = SafeTensors.deserialize(joinpath(fixture(name), "acts.safetensors")).metadata
        model = fixture_model(fixture(name))
        @test sum(length, Flux.trainables(model)) == parse(Int, meta["num_params"])
    end
end

@testset "gradient" begin
    for name in MODERNBERT_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        args, kwargs = fixture_inputs(fixture_acts(name))
        @test check_gradient(model, args...; kwargs...)
    end
end

@testset "half precision" begin
    for name in MODERNBERT_FIXTURES
        @test check_half_precision(name)
    end
end

const TINY = (;
    vocab_size = 20, hidden_size = 8, intermediate_size = 6, num_hidden_layers = 4,
    num_attention_heads = 2, max_position_embeddings = 32, local_attention = 4)

@testset "ModernBertModel" begin
    @test issetequal(
        ["ModernBertModel", "ModernBertForMaskedLM",
            "ModernBertForSequenceClassification"],
        list_models(; byfamily = true)["modernbert"])
    model = create_model("ModernBertModel"; TINY...)
    @test model isa ModernBertModel
    # one global layer in three, the first included; no norm before the first attention
    types = [l.attn.layer_type for l in model.layers]
    @test types ==
          [:full_attention, :sliding_attention, :sliding_attention, :full_attention]
    @test [l.attn.sliding_window for l in model.layers] == [nothing, 3, 3, nothing]
    @test model.layers[1].attn_norm === identity
    @test model.layers[2].attn_norm isa BiaslessLayerNorm &&
          model.final_norm isa BiaslessLayerNorm
    @test keys(model.rotary_emb) == (:full_attention, :sliding_attention)
    @test model.rotary_emb.full_attention.rope_parameters.rope_theta == 160_000
    @test model.rotary_emb.sliding_attention.rope_parameters.rope_theta == 10_000
    @test model.layers[1].attn.Wqkv.bias === false && model.layers[1].mlp.Wi.bias === false

    ids = rand(1:20, 12, 3)
    y = model(ids)
    @test size(y) == (8, 12, 3)
    @test size(model(ids[:, 1])) == (8, 12)
    @test model(ids[:, 2]) ≈ y[:, :, 2]

    # padding on the right leaves the other tokens unchanged, and long sequences too,
    # beyond the window of the local layers
    mask = Int.((1:12) .<= [12 7 3])
    padded = model(ids; attention_mask = mask)
    @test padded[:, :, 1] ≈ y[:, :, 1]
    @test padded[:, 1:7, 2] ≈ model(ids[1:7, 2])
    @test padded[:, 1:3, 3] ≈ model(ids[1:3, 3])
    @test all(isfinite, padded)
    @test model(ids[:, 2]; attention_mask = mask[:, 2]) ≈
          padded[:, :, 2]
    @test_throws "pad on the right" model(ids; attention_mask = reverse(mask; dims = 1))
    @test_throws "token_type_ids" model(ids; token_type_ids = zeros(Int, 12, 3))

    # the local layers see the tokens within 2 of each, the global ones all
    other = copy(ids)
    other[12, 1] = mod1(ids[12, 1] + 1, 20)
    changed = model(other)[:, :, 1] .!= y[:, :, 1]
    @test any(changed[:, 1])  # the first token sees the last through the global layers
    local_only = create_model("ModernBertModel"; TINY..., num_hidden_layers = 2,
        layer_types = fill("sliding_attention", 2))
    a, b = local_only(ids[:, 1]), local_only(other[:, 1])
    # two local layers see 2 + 2 tokens away: the tokens 8 to 12 change
    @test a[:, 1:7] ≈ b[:, 1:7]
    @test a[:, 8] != b[:, 8]  # by a little: the weights are small

    # the features, the intermediates and the output of a processor
    @test forward_features(model, ids) ≈ y
    features, hidden = forward_intermediates(model, ids; attention_mask = mask)
    @test features ≈ padded && length(hidden) == 4
    @test model.final_norm(hidden[end]) ≈ features
    normed = forward_intermediates(model, ids; norm = true, indices = [4],
        intermediates_only = true)
    @test only(normed) ≈ y
    @test feature_info(model) == [(; num_chs = 8, reduction = 1, module_name = "layers.$i")
           for i in 0:3]
    @test model((; input_ids = ids, attention_mask = mask)) ≈
          padded
    @test_throws "has no `input_ids`" model((; attention_mask = mask))

    # configurations: the keys of the older configurations
    legacy = create_model("ModernBertModel"; TINY..., global_attn_every_n_layers = 2,
        global_rope_theta = 5.0e4, local_rope_theta = 50.0)
    @test [l.attn.layer_type for l in legacy.layers] ==
          [:full_attention, :sliding_attention, :full_attention, :sliding_attention]
    @test legacy.rotary_emb.full_attention.rope_parameters.rope_theta == 5.0e4
    @test legacy.rotary_emb.sliding_attention.rope_parameters.rope_theta == 50
    nested = create_model("ModernBertModel"; TINY...,
        rope_parameters = Dict(
            "full_attention" => Dict("rope_type" => "linear",
                "rope_theta" => 1000.0, "factor" => 2.0),
            "sliding_attention" => Dict("rope_type" => "default", "rope_theta" => 7.0)))
    @test nested.rotary_emb.full_attention.rope_parameters ==
          (; rope_type = "linear", rope_theta = 1000.0, factor = 2.0)
    @test nested.rotary_emb.sliding_attention.rope_parameters.rope_theta == 7
    @test_throws "rope_type" create_model("ModernBertModel"; TINY...,
        rope_scaling = Dict("rope_type" => "yarn", "factor" => 2.0))
    @test_throws "entries" create_model("ModernBertModel"; TINY..., layer_types = ["full_attention"])
    @test_throws "layer_types" create_model("ModernBertModel"; TINY...,
        layer_types = fill("chunked", 4))
    @test_throws "hidden_activation" create_model("ModernBertModel"; TINY...,
        hidden_activation = "foo")
    @test_throws "divisible" create_model("ModernBertModel"; TINY..., num_attention_heads = 3)
    biased = create_model("ModernBertModel"; TINY..., norm_bias = true)
    @test biased.final_norm isa Flux.LayerNorm
    @test Metalhead.find_format(Dict("model_type" => "modernbert")) ==
          (Metalhead.Transformers.TRANSFORMERS, "ModernBertModel")
end

@testset "ModernBertForMaskedLM" begin
    model = create_model("ModernBertForMaskedLM"; TINY...)
    @test model isa ModernBertForMaskedLM
    ids = rand(1:20, 5, 2)
    logits = model(ids)
    @test size(logits) == (20, 5, 2)
    hidden = forward_features(model, ids)
    @test hidden ≈ model.model(ids)
    @test logits ≈ forward_head(model, hidden)
    # the decoder shares the weight of the token embeddings
    @test model.decoder.weight === model.model.embeddings.tok_embeddings.weight
    expected = transpose(model.decoder.weight) * reshape(model.head(hidden), 8, :) .+
               model.decoder.bias
    @test logits ≈ reshape(expected, 20, 5, 2)
    @test model(ids[:, 1]) ≈ logits[:, :, 1]
    untied = create_model("ModernBertForMaskedLM"; TINY..., tie_word_embeddings = false,
        decoder_bias = false)
    @test untied.decoder isa Flux.Dense && untied.decoder.bias === false
    @test feature_info(model)[1].module_name == "model.layers.0"
end

@testset "ModernBertForSequenceClassification" begin
    ids = rand(1:20, 6, 3)
    mask = [1 1 1; 1 1 1; 1 1 1; 1 1 1; 0 1 1; 0 0 1]
    model = create_model("ModernBertForSequenceClassification"; TINY..., num_labels = 3)
    @test model isa ModernBertForSequenceClassification
    logits = model(ids; attention_mask = mask)
    @test size(logits) == (3, 3)
    hidden = model.model(ids; attention_mask = mask)
    @test logits ≈ model.classifier(model.head(hidden[:, 1, :]))
    @test model(ids[1:4, 1]) ≈ model(ids; attention_mask = mask)[:, 1]
    @test model((; input_ids = ids, attention_mask = mask)) ≈ logits

    # the mean of the tokens that the mask keeps
    mean_model = create_model(
        "ModernBertForSequenceClassification"; TINY..., num_labels = 3,
        classifier_pooling = "mean")
    mean_hidden = mean_model.model(ids; attention_mask = mask)
    pooled = reduce(hcat, [sum(mean_hidden[:, 1:n, i]; dims = 2) ./ n
                           for (i, n) in enumerate((4, 5, 6))])
    @test mean_model(ids; attention_mask = mask) ≈
          mean_model.classifier(mean_model.head(pooled))
    @test mean_model(ids[1:4, 1]) ≈ mean_model(ids; attention_mask = mask)[:, 1]
    # without a mask, the mean of all tokens
    @test forward_head(mean_model, mean_hidden; pre_logits = true) ≈
          mean_model.head(dropdims(sum(mean_hidden; dims = 2); dims = 2) ./ 6)
    @test_throws "classifier_pooling" create_model("ModernBertForSequenceClassification";
        TINY..., classifier_pooling = "max")
    @test size(create_model("ModernBertForSequenceClassification"; TINY...)(ids)) == (2, 3)
end

if TEST_NETWORK
    # the tokenizer of the processor, which METALHEAD_TEST_NETWORK adds to the test
    # environment (runtests.jl)
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers
    @testset "references" begin
        for arch in ("ModernBertModel", "ModernBertForMaskedLM",
            "ModernBertForSequenceClassification")
            for file in reference_files_of(arch)
                r = check_encoder_reference(file)
                @test r.inputs
                @test r.hidden <= 1e-4
                @test r.pooled <= 1e-4
                @test r.logits <= 1e-4
            end
        end
    end
end
