using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup
using Flux: Flux
using Metalhead.Transformers: BertModel, BertForSequenceClassification

const BERT_FIXTURES = ["bert_tiny", "bert_options", "bert_classifier"]

fixture_acts(name) = Metalhead.Checkpoint(joinpath(fixture(name), "acts.safetensors"))

@testset "fixtures" begin
    for name in BERT_FIXTURES
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
    for name in BERT_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        args, kwargs = fixture_inputs(fixture_acts(name))
        has_pooler = hasproperty(model, :pooler) && model.pooler !== nothing
        model = has_pooler ? WithPooler(model) : model
        @test check_gradient(model, args...; kwargs...)
    end
end

@testset "half precision" begin
    for name in BERT_FIXTURES
        @test check_half_precision(name)
    end
end

const TINY = (; vocab_size = 20, hidden_size = 8, num_hidden_layers = 2,
    num_attention_heads = 2, intermediate_size = 12, max_position_embeddings = 10)

@testset "BertModel" begin
    @test issetequal(["BertModel", "BertForSequenceClassification"],
        list_models(; byfamily = true)["bert"])
    model = create_model("BertModel"; TINY...)
    @test model isa BertModel
    ids = rand(1:20, 6, 3)
    y = model(ids)
    @test size(y) == (8, 6, 3)
    # the pooler: the first token through a dense layer and tanh
    pooled = forward_head(model, y)
    @test size(pooled) == (8, 3) && pooled ≈ tanh.(model.pooler.dense(y[:, 1, :]))

    # a vector is one sequence, without a batch dimension
    @test model(ids[:, 2]) ≈ y[:, :, 2]
    @test forward_head(model, model(ids[:, 2])) ≈ pooled[:, 2]

    # token types, 0-based segment ids, default to 0
    zeros_types = zeros(Int, 6, 3)
    @test model(ids; token_type_ids = zeros_types) ≈ y
    types = [0 0 1; 0 1 1; 1 1 1; 1 1 0; 1 0 0; 1 0 0]
    @test !(model(ids; token_type_ids = types) ≈ y)
    @test model(ids[:, 3]; token_type_ids = types[:, 3]) ≈
          model(ids; token_type_ids = types)[:, :, 3]
    @test_throws "token_type_ids` has size" model(ids; token_type_ids = types[1:5, :])

    # padding on the right leaves the other tokens unchanged, and the pooled one too
    mask = [1 1 1; 1 1 1; 1 1 1; 1 1 1; 0 1 1; 0 0 1]
    padded = model(ids; attention_mask = mask, token_type_ids = types)
    @test padded[:, 1:4, 1] ≈ model(ids[1:4, 1]; token_type_ids = types[1:4, 1])
    @test padded[:, 1:5, 2] ≈ model(ids[1:5, 2]; token_type_ids = types[1:5, 2])
    @test forward_head(model, padded)[:, 1] ≈
          forward_head(model, model(ids[1:4, 1]; token_type_ids = types[1:4, 1]))
    @test padded[:, :, 3] ≈ model(ids[:, 3]; token_type_ids = types[:, 3])
    @test model(ids[:, 1]; attention_mask = mask[:, 1], token_type_ids = types[:, 1]) ≈
          padded[:, :, 1]
    @test_throws "pad on the right" model(ids; attention_mask = reverse(mask; dims = 1))
    @test_throws "size" model(ids; attention_mask = mask[1:5, :])
    @test_throws "max_position_embeddings = 10" model(rand(1:20, 11, 1))

    # the features and the output of each layer
    @test forward_features(model, ids) ≈ y
    @test forward_features(model, ids[:, 1]) ≈ y[:, :, 1]
    features, hidden = forward_intermediates(model, ids; token_type_ids = types)
    @test features ≈ model(ids; token_type_ids = types)
    @test length(hidden) == 2 && hidden[end] ≈ features && size(hidden[1]) == (8, 6, 3)
    @test feature_info(model) == [(; num_chs = 8, reduction = 1,
               module_name = "encoder.layer.$i") for i in 0:1]
    @test only(forward_intermediates(model, ids; token_type_ids = types, indices = [1],
        stop_early = true, intermediates_only = true)) ≈ hidden[1]
    @test forward_intermediates(model, ids; token_type_ids = types, indices = 1,
        intermediates_only = true) ≈ hidden[2:2]
    @test forward_intermediates(model, ids[:, 2]; token_type_ids = types[:, 2],
        indices = [1], intermediates_only = true)[1] ≈ hidden[1][:, :, 2]
    @test_throws "indices" forward_intermediates(model, ids; indices = [3])

    # the output of a processor, as it is
    inputs = (; input_ids = ids, attention_mask = mask, token_type_ids = types)
    @test model(inputs) ≈ padded
    @test forward_features(model, inputs) ≈ padded
    @test forward_intermediates(model, inputs; indices = [1],
        intermediates_only = true)[1] ≈
          forward_intermediates(model, ids; attention_mask = mask, token_type_ids = types,
        indices = [1], intermediates_only = true)[1]
    @test model((; input_ids = ids)) ≈ y
    @test_throws "has no `input_ids` (it has `attention_mask`)" model((;
        attention_mask = mask))
    @test_throws "pass `text` to the processor" model((; attention_mask = mask))

    # configurations
    nopool = create_model("BertModel"; TINY..., add_pooling_layer = false)
    @test nopool.pooler === nothing
    @test_throws "without a pooler" forward_head(nopool, nopool(ids))
    for act in ("gelu", "gelu_new", "gelu_pytorch_tanh", "relu", "silu", "swish", "quick_gelu")
        @test size(create_model("BertModel"; TINY..., hidden_act = act)(ids)) == (8, 6, 3)
    end
    @test_throws "hidden_act" create_model("BertModel"; TINY..., hidden_act = "gelu_fast")
    @test_throws "is_decoder" create_model("BertModel"; TINY..., is_decoder = true)
    @test_throws "add_cross_attention" create_model("BertModel"; TINY...,
        add_cross_attention = true)
    @test_throws "divisible" create_model("BertModel"; TINY..., num_attention_heads = 3)
    # a config.json without `architectures` names the model by its `model_type`
    @test Metalhead.find_format(Dict("model_type" => "bert")) ==
          (Metalhead.Transformers.TRANSFORMERS, "BertModel")
end

@testset "BertForSequenceClassification" begin
    ids = rand(1:20, 6, 3)
    types = [0 0 1; 0 1 1; 1 1 1; 1 1 0; 1 0 0; 1 0 0]
    mask = [1 1 1; 1 1 1; 1 1 1; 1 1 1; 0 1 1; 0 0 1]
    model = create_model("BertForSequenceClassification"; TINY..., num_labels = 3)
    @test model isa BertForSequenceClassification
    logits = model(ids)
    @test size(logits) == (3, 3)
    pooled = forward_head(model.bert, model.bert(ids))
    @test logits ≈ model.classifier(pooled)
    @test forward_head(model, forward_features(model, ids)) ≈ logits
    @test forward_head(model, forward_features(model, ids); pre_logits = true) ≈ pooled
    @test model(ids[:, 2]) ≈ logits[:, 2]
    @test model(ids; attention_mask = mask, token_type_ids = types) ≈
          model((; input_ids = ids, attention_mask = mask, token_type_ids = types))
    @test model(ids[:, 1]; attention_mask = mask[:, 1], token_type_ids = types[:, 1]) ≈
          model(ids; attention_mask = mask, token_type_ids = types)[:, 1]
    @test feature_info(model)[1].module_name == "bert.encoder.layer.0"
    features, hidden = forward_intermediates(model, ids)
    @test features ≈ forward_features(model, ids) && length(hidden) == 2

    # the number of labels: `id2label`, else `num_labels`, else 2
    labels(; kwargs...) = size(
        create_model("BertForSequenceClassification"; TINY...,
            kwargs...).classifier.weight,
        1)
    @test labels() == 2
    @test labels(; num_labels = 5) == 5
    @test labels(; id2label = (; var"0" = "a", var"1" = "b", var"2" = "c")) == 3
    @test labels(; id2label = Dict("0" => "a"), num_labels = 1) == 1
    @test_throws "incompatible" labels(; id2label = Dict("0" => "a"), num_labels = 2)
    @test_throws "is_decoder" labels(; is_decoder = true)
end

if TEST_NETWORK
    # the tokenizer of the processor, which METALHEAD_TEST_NETWORK adds to the test
    # environment (runtests.jl)
    @eval using HuggingFaceTokenizers: HuggingFaceTokenizers
    @testset "references" begin
        for arch in ("BertModel", "BertForSequenceClassification")
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
