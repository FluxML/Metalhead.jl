using Metalhead, Test
using Flux: Flux
include("../parity_setup.jl")
using .ParitySetup

const QWEN3_5_FIXTURES = ["qwen3_5_text_tiny", "qwen3_5_text_options", "qwen3_5_text_padded"]
const chunk_gated_delta_rule = Metalhead.Transformers.chunk_gated_delta_rule

@testset "fixtures" begin
    for name in QWEN3_5_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "gradient" begin
    for name in QWEN3_5_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        vocab_size = size(model.model.embed_tokens.weight, 2)
        @test check_gradient(model, rand(1:vocab_size, 5, 2))
    end
    # through the state carried from a chunk of 64 tokens to the next
    model = fixture_model(fixture("qwen3_5_text_padded"); eltype = Float32)
    @test check_gradient(model, rand(1:32, 70, 2))
end

@testset "half precision" begin
    # These random models amplify the rounding of each layer, and Float16 matrix products
    # accumulate in 16 bits on the CPU: Float16 errs by 1.6 to 4.3% here, PyTorch's float16
    # (accumulating in float32) by 0.6 to 1.1% and its bfloat16 by 8 to 12%.
    for name in QWEN3_5_FIXTURES
        @test check_half_precision(name; rtol = 6e-2)
    end
end

# transformers' `torch_recurrent_gated_delta_rule`, token by token: the sum of the outputs
# (dv, heads, n, batch) multiplied by `w`.
function recurrent_delta_rule(q, k, v, g, beta, w)
    dk, h, n, b = size(q)
    dv = size(v, 1)
    l2norm(x) = x ./ sqrt.(sum(abs2, x; dims = 1) .+ 1e-6)
    q, k = l2norm(q) ./ sqrt(dk), l2norm(k)
    s = zeros(dv, dk, h, b)
    total = 0.0
    for t in 1:n
        qt, kt = reshape(q[:, :, t, :], 1, dk, h, b), reshape(k[:, :, t, :], 1, dk, h, b)
        s = s .* reshape(exp.(g[:, t, :]), 1, 1, h, b)
        delta = (reshape(v[:, :, t, :], dv, 1, h, b) .- sum(s .* kt; dims = 2)) .*
                reshape(beta[:, t, :], 1, 1, h, b)
        s = s .+ delta .* kt
        total += sum(dropdims(sum(s .* qt; dims = 2); dims = 2) .* w[:, :, t, :])
    end
    return total
end

@testset "gated delta rule" begin
    dk, dv, h, n, b = 4, 3, 2, 70, 2
    q, k, v = randn(dk, h, n, b), randn(dk, h, n, b), randn(dv, h, n, b)
    g, beta = -0.3 .* rand(h, n, b), rand(h, n, b)
    w = randn(dv, h, n, b)
    chunked(chunk_size) = (q, k, v, g, beta) -> sum(w .* chunk_gated_delta_rule(q, k, v,
        g, beta; chunk_size))
    reference = recurrent_delta_rule(q, k, v, g, beta, w)
    @test chunked(64)(q, k, v, g, beta) ≈ reference rtol = 1e-12
    @test chunk_gated_delta_rule(q, k, v, g, beta; chunk_size = 16) ≈
          chunk_gated_delta_rule(q, k, v, g, beta) rtol = 1e-12
    expected = Flux.gradient((q, k, v, g, beta) -> recurrent_delta_rule(q, k, v, g, beta, w),
        q, k, v, g, beta)
    for chunk_size in (16, 64)
        grads = Flux.gradient(chunked(chunk_size), q, k, v, g, beta)
        @test all(map((a, b) -> isapprox(a, b; rtol = 1e-10), grads, expected))
    end
    y = chunk_gated_delta_rule(Float32.(q), Float32.(k), Float16.(v), Float32.(g),
        Float32.(beta))
    @test eltype(y) == Float16
end

@testset "config" begin
    tiny = (; vocab_size = 16, hidden_size = 8, intermediate_size = 12,
        num_hidden_layers = 4, num_attention_heads = 2, num_key_value_heads = 1,
        head_dim = 8, linear_num_key_heads = 1, linear_num_value_heads = 2,
        linear_key_head_dim = 4, linear_value_head_dim = 4)
    qwen3_5(; kwargs...) = create_model("Qwen3_5ForCausalLM"; tiny..., kwargs...)
    @test "Qwen3_5ForCausalLM" in list_models(; byfamily = true)["qwen"]
    @test Metalhead.lookup("Qwen3_5ForCausalLM").ignore == [r"^mtp\."]
    model = qwen3_5()
    @test size(model(rand(1:16, 5, 3))) == (16, 5, 3)
    @test size(model(rand(1:16, 5))) == (16, 5)

    # every `full_attention_interval`-th layer has full attention, unless `layer_types` says
    L, F = "linear_attention", "full_attention"
    layer_types(m) = [l.self_attn === nothing ? L : F for l in m.model.layers]
    @test layer_types(model) == [L, L, L, F]
    @test layer_types(qwen3_5(; full_attention_interval = 2)) == [L, F, L, F]
    @test layer_types(qwen3_5(; layer_types = [F, L, F, L])) == [F, L, F, L]
    layer = model.model.layers[1]
    @test layer.linear_attn isa Metalhead.Transformers.Qwen3_5GatedDeltaNet
    @test size(layer.linear_attn.conv1d.weight) == (4, 1, 16)
    attn = model.model.layers[4].self_attn
    @test size(attn.q_proj.weight) == (32, 8)  # queries and gates
    @test attn.q_proj.bias === false
    @test qwen3_5(; attention_bias = true).model.layers[4].self_attn.q_proj.bias isa Vector
    @test model.lm_head isa Flux.Dense
    tied = qwen3_5(; tie_word_embeddings = true)
    @test tied.lm_head.weight === tied.model.embed_tokens.weight

    # the rotary embedding of `rope_parameters`, or of the keys of older configs
    rope = model.model.rotary_emb
    @test (rope.rotary_dim, rope.rope_theta, rope.mrope_section) == (2, 10_000.0, (11, 11, 10))
    rope = qwen3_5(; rope_parameters = Dict("rope_type" => "default", "rope_theta" => 1e6,
        "partial_rotary_factor" => 0.5, "mrope_section" => [1, 1, 0])).model.rotary_emb
    @test (rope.rotary_dim, rope.rope_theta, rope.mrope_section) == (4, 1e6, (1, 1, 0))
    rope = qwen3_5(; rope_theta = 5.0, partial_rotary_factor = 1.0).model.rotary_emb
    @test (rope.rotary_dim, rope.rope_theta) == (8, 5.0)

    @test_throws "layer_types" qwen3_5(; layer_types = [F, L])
    @test_throws "sliding_attention" qwen3_5(; layer_types = [L, L, L, "sliding_attention"])
    @test_throws "rope_type" qwen3_5(;
        rope_parameters = Dict("rope_type" => "yarn", "factor" => 4.0))
    @test_throws "partial_rotary_factor" qwen3_5(; partial_rotary_factor = 0.375)
    @test_throws "linear_num_value_heads" qwen3_5(; linear_num_key_heads = 2,
        linear_num_value_heads = 3)
    @test_throws "hidden_act" qwen3_5(; hidden_act = "relu")

    # the positions (temporal, height, width) of the multimodal rotary embedding, by default
    # those of text; a first row of four, transformers' text positions, is ignored
    n = 70
    ids = rand(1:16, n, 2)
    text = repeat(reshape(0:(n - 1), 1, n, 1), 3)
    @test model(ids; position_ids = text) ≈ model(ids)
    @test model(ids; position_ids = [fill(7, 1, n, 1); text]) ≈ model(ids)
    pos = rand(0:100, 3, n, 2)
    batch = model(ids; position_ids = pos)
    @test !(batch ≈ model(ids))
    @test batch[:, :, 2] ≈ model(ids[:, 2]; position_ids = pos[:, :, 2])
    @test_throws "position_ids" model(ids; position_ids = pos[:, 1:5, :])
    embeds = model.model.embed_tokens(ids)
    @test Metalhead.Transformers.qwen3_5_decode(model.model, embeds, pos) ≈
          forward_features(model, ids; position_ids = pos)
    @test only(forward_intermediates(model, ids; indices = [4], intermediates_only = true,
        position_ids = pos)) ≈ last(forward_intermediates(model, ids; position_ids = pos)[2])

    # padding on the right, across chunks of 64 tokens, leaves the other tokens unchanged
    mask = [ones(Int, n) [ones(Int, 66); zeros(Int, 4)]]
    padded = model(ids; attention_mask = mask)
    @test padded[:, :, 1] ≈ model(ids[:, 1])
    @test padded[:, 1:66, 2] ≈ model(ids[1:66, 2])
    @test model(ids[:, 2]; attention_mask = mask[:, 2]) ≈ padded[:, :, 2]
    @test_throws "pad on the right" model(ids; attention_mask = reverse(mask; dims = 1))
    @test_throws "size" model(ids; attention_mask = mask[1:5, :])
    hidden = forward_intermediates(model, ids; indices = [1], intermediates_only = true,
        attention_mask = mask)
    @test only(hidden)[:, 1:66, 2] ≈
          only(forward_intermediates(model, ids[1:66, 2:2]; indices = [1],
        intermediates_only = true))[:, :, 1]
end

@testset "vision-language model" begin
    r = check_fixture(fixture("qwen3_5_tiny"))
    @test r.output <= 1e-8
    @test r.features <= 1e-8
    @test r.pass
    @test check_half_precision("qwen3_5_tiny"; rtol = 6e-2)

    model = fixture_model(fixture("qwen3_5_tiny"); eltype = Float32)
    (ids,), inputs = fixture_inputs(Metalhead.Checkpoint(joinpath(fixture("qwen3_5_tiny"),
        "acts.safetensors")))
    pixel_values, image_grid_thw = Float32.(inputs.pixel_values), inputs.image_grid_thw
    @test check_gradient(model, ids; pixel_values, image_grid_thw, inputs.attention_mask)
    @test "Qwen3_5ForConditionalGeneration" in list_models(; byfamily = true)["qwen"]

    # each sequence of the batch, with its own image, alone
    batch = model(ids; pixel_values, image_grid_thw, inputs.attention_mask)
    first_patches = sum(prod, eachcol(image_grid_thw[:, 1:2]))
    alone = model(ids[1:24, 2]; pixel_values = pixel_values[:, (first_patches + 1):end],
        image_grid_thw = image_grid_thw[:, 3:3])
    @test alone ≈ batch[:, 1:24, 2]
    # text alone is the text model's
    text = rand(1:59, 9, 2)
    @test model(text) ≈ model.lm_head(model.model.language_model(text))
    @test forward_features(model, text[:, 1]) ≈ model.model.language_model(text[:, 1])
    hidden = forward_intermediates(model, ids; pixel_values, image_grid_thw,
        inputs.attention_mask)
    @test first(hidden) ≈ forward_features(model, ids; pixel_values, image_grid_thw,
        inputs.attention_mask)

    # transformers' `get_rope_index`: the merged 4 × 6 grid of patches at positions 3:8 takes
    # the positions of its 2 × 3 grid from 3, and the text after it goes on from 3 + 3
    image = 61  # `image_token_id + 1`
    tokens = [1, 2, 3, fill(image, 6)..., 4, 5]
    positions = Metalhead.Transformers.rope_index(tokens[:, :], [1; 4; 6;;], image, 2,
        nothing)
    @test positions[:, :, 1] == [0 1 2 3 3 3 3 3 3 6 7
                                 0 1 2 3 3 3 4 4 4 6 7
                                 0 1 2 3 4 5 3 4 5 6 7]
    @test_throws "image tokens" model(ids[1:10, :]; pixel_values, image_grid_thw)
    @test_throws "image_grid_thw" model(ids; pixel_values)
end

@testset "processor inputs" begin
    # the output of a Qwen3VLProcessor, as it is, with the image tokens of the tiny model
    # (its own tests are in test/processors/qwen3_vl.jl)
    T = Metalhead.Transformers
    model = fixture_model(fixture("qwen3_5_tiny"); eltype = Float32)
    tok = T.TokenizersBackend(; pad_token = "<pad>",
        added_tokens_decoder = Dict(0 => "<pad>", 60 => "<|image_pad|>",
            61 => "<|video_pad|>"))
    ip = T.Qwen2VLImageProcessor(; patch_size = 2, image_mean = (0.5, 0.5, 0.5),
        image_std = (0.5, 0.5, 0.5), size = (; shortest_edge = 16, longest_edge = 256))
    proc = T.Qwen3VLProcessor(ip, tok)
    images = [rand(UInt8, 12, 8, 3), rand(UInt8, 9, 7, 3)]
    prompt = [5, 62, 60, 63, 7, 8]  # <|vision_start|><|image_pad|><|vision_end|>
    one = proc(; images = images[1], text = prompt)
    @test count(==(61), one.input_ids) == 6  # (8 ÷ 2) × (12 ÷ 2) ÷ 2²
    logits = model(one)
    @test size(logits) == (64, length(one.input_ids))  # logits[:, end] for the next token
    @test logits ≈ model(one.input_ids; one.pixel_values, one.image_grid_thw)
    @test model.model(one) ≈ forward_features(model, one.input_ids; one.pixel_values,
        one.image_grid_thw)
    # a batch, padded on the right: each prompt as alone
    batch = proc(; images, text = [prompt, [62, 60, 63, 9, 10, 11, 12, 13, 14]])
    n = length(one.input_ids)
    @test size(batch.input_ids) == (n + 1, 2)
    @test batch.attention_mask == [ones(Int, n) ones(Int, n); 0 1]
    y = model(batch)
    @test y[:, 1:n, 1] ≈ logits
    second = proc(; images = images[2], text = [62, 60, 63, 9, 10, 11, 12, 13, 14])
    @test y[:, :, 2] ≈ model(second)
    # text alone, and inputs of another model
    @test model((; input_ids = [1, 2, 3])) ≈ model([1, 2, 3])
    @test_throws "has no `input_ids`" model((; one.pixel_values, one.image_grid_thw))
    @test_throws "has no `input_ids`" model.model((; one.pixel_values))
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files_of("Qwen3_5ForConditionalGeneration")
        for r in (check_lm_reference(file), check_image_reference(file))
            @test r.hidden_states <= 1e-4
            @test r.last_logits <= 1e-4
            @test r.generated
        end
    end
end
