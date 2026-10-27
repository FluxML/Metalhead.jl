using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup
using Flux: Flux
using Metalhead.Transformers: XLMRobertaModel, XLMRobertaForSequenceClassification

const XLM_ROBERTA_FIXTURES = [
    "xlm_roberta_tiny", "xlm_roberta_options", "xlm_roberta_classifier"]

fixture_acts(name) = Metalhead.Checkpoint(joinpath(fixture(name), "acts.safetensors"))

@testset "fixtures" begin
    for name in XLM_ROBERTA_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
        # the parameters of transformers' model, all loaded by name
        meta = SafeTensors.deserialize(joinpath(fixture(name), "acts.safetensors")).metadata
        model = fixture_model(fixture(name))
        @test sum(length, Flux.trainables(model)) == parse(Int, meta["num_params"])
        # the pooler output of a base model with a pooler
        acts = fixture_acts(name)
        if haskey(acts, "pooler_output")
            args, kwargs = fixture_inputs(acts)
            pooled = forward_head(model, forward_features(model, args...; kwargs...))
            @test relerr(pooled, acts["pooler_output"]) <= 1e-8
        end
    end
end

# The model with the pooler's output too, whose parameters the hidden states don't reach
struct WithPooler{M}
    model::M
end
Flux.@layer WithPooler
function (m::WithPooler)(args...; kwargs...)
    x = m.model(args...; kwargs...)
    return (; last_hidden_state = x, pooler_output = forward_head(m.model, x))
end

@testset "gradient" begin
    for name in XLM_ROBERTA_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        args, kwargs = fixture_inputs(fixture_acts(name))
        has_pooler = hasproperty(model, :pooler) && model.pooler !== nothing
        model = has_pooler ? WithPooler(model) : model
        @test check_gradient(model, args...; kwargs...)
    end
end

@testset "half precision" begin
    for name in XLM_ROBERTA_FIXTURES
        @test check_half_precision(name)
    end
end

const TINY = (; vocab_size = 20, hidden_size = 8, num_hidden_layers = 2,
    num_attention_heads = 2, intermediate_size = 12, max_position_embeddings = 10,
    type_vocab_size = 1, pad_token_id = 1)

@testset "XLMRobertaModel" begin
    @test issetequal(["XLMRobertaModel", "XLMRobertaForSequenceClassification"],
        list_models(; byfamily = true)["xlm_roberta"])
    model = create_model("XLMRobertaModel"; TINY...)
    @test model isa XLMRobertaModel
    # token indices are 1-based: the padding token 1 is the index 2
    ids = [3 5 7; 8 9 10; 11 12 2; 13 2 2; 2 2 2; 2 2 2]
    mask = Int.(ids .!= 2)
    y = model(ids; attention_mask = mask)
    @test size(y) == (8, 6, 3)
    @test forward_head(model, y) ≈ tanh.(model.pooler.dense(y[:, 1, :]))

    # the positions count the tokens that are not padding, from pad_token_id + 1 = 2
    @test Metalhead.Transformers.position_indices(ids, 1) ==
          [3 3 3; 4 4 4; 5 5 2; 6 2 2; 2 2 2; 2 2 2]
    # so padding changes nothing of the other tokens, with or without the mask
    @test y[:, 1:4, 1] ≈ model(ids[1:4, 1])
    @test y[:, 1:2, 3] ≈ model(ids[1:2, 3])
    @test forward_head(model, y)[:, 1] ≈ forward_head(model, model(ids[1:4, 1]))
    @test model(ids[:, 2]; attention_mask = mask[:, 2]) ≈
          y[:, :, 2]

    # sequences hold up to max_position_embeddings - pad_token_id - 1 tokens
    @test size(model(rand(3:20, 8))) == (8, 8)
    @test_throws "up to 8 tokens" model(rand(3:20, 9))
    @test size(model(vcat(rand(3:20, 8), [2, 2]))) == (8, 10)

    # token types are 0-based segment ids: XLM-RoBERTa has one, as its checkpoints
    @test size(model(ids; token_type_ids = zeros(Int, 6, 3))) == (8, 6, 3)
    two_types = create_model("XLMRobertaModel"; TINY..., type_vocab_size = 2)
    types = [0 0 1; 0 1 1; 1 1 1; 1 1 0; 1 0 0; 1 0 0]
    @test !(two_types(ids; token_type_ids = types) ≈
            two_types(ids))

    # the features, the intermediates and the output of a processor
    @test forward_features(model, ids; attention_mask = mask) ≈ y
    features, hidden = forward_intermediates(model, ids; attention_mask = mask)
    @test features ≈ y && length(hidden) == 2
    @test feature_info(model)[2].module_name == "encoder.layer.1"
    @test model((; input_ids = ids, attention_mask = mask)) ≈
          y
    @test_throws "has no `input_ids`" model((; attention_mask = mask))

    @test_throws "pad_token_id" create_model("XLMRobertaModel"; TINY..., pad_token_id = nothing)
    @test_throws "is_decoder" create_model("XLMRobertaModel"; TINY..., is_decoder = true)
    @test create_model("XLMRobertaModel"; TINY..., add_pooling_layer = false).pooler ===
          nothing
    @test Metalhead.find_format(Dict("model_type" => "xlm-roberta")) ==
          (Metalhead.Transformers.TRANSFORMERS, "XLMRobertaModel")
end

@testset "XLMRobertaForSequenceClassification" begin
    model = create_model("XLMRobertaForSequenceClassification"; TINY..., num_labels = 1)
    @test model isa XLMRobertaForSequenceClassification && model.roberta.pooler === nothing
    ids = [3 5 7; 8 9 10; 11 12 2; 13 2 2; 2 2 2; 2 2 2]
    mask = Int.(ids .!= 2)
    logits = model(ids; attention_mask = mask)
    @test size(logits) == (1, 3)
    # a dense layer and tanh on the first token, then a linear layer
    first_token = model.roberta(ids; attention_mask = mask)[:, 1, :]
    @test logits ≈ model.classifier.out_proj(tanh.(model.classifier.dense(first_token)))
    @test forward_head(model, forward_features(model, ids; attention_mask = mask)) ≈ logits
    @test forward_head(model, forward_features(model, ids; attention_mask = mask);
        pre_logits = true) ≈ tanh.(model.classifier.dense(first_token))
    @test model(ids[1:4, 1]) ≈ logits[:, 1]
    @test model((; input_ids = ids, attention_mask = mask)) ≈ logits
    @test feature_info(model)[1].module_name == "roberta.encoder.layer.0"
    @test size(create_model("XLMRobertaForSequenceClassification"; TINY...)(ids)) == (2, 3)
end

if TEST_NETWORK
    # the tokenizer of the processor, which METALHEAD_TEST_NETWORK adds to the test
    # environment (runtests.jl)
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers
    @testset "references" begin
        for arch in ("XLMRobertaModel", "XLMRobertaForSequenceClassification")
            for file in reference_files_of(arch)
                # BGE-reranker-v2-m3 has 568M parameters: 2.2 GB to download
                occursin("bge-reranker-v2-m3", file) && !TEST_LARGE && continue
                r = check_encoder_reference(file)
                @test r.inputs
                @test r.hidden <= 1e-4
                @test r.pooled <= 1e-4
                @test r.logits <= 1e-4
            end
        end
    end
end
