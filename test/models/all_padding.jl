using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup
using Metalhead.Transformers: query_lengths

# A sequence of a batch with no token (all padding) has every key masked. PyTorch's
# attention (sdpa) gives zeros there, and so finite outputs; NNlib's softmax gave NaNs.

@testset "query_lengths" begin
    @test query_lengths(nothing, 5) === nothing
    @test query_lengths(nothing, 5, 3) === nothing
    @test query_lengths([5, 2], 5) === nothing
    @test query_lengths([5, 0, 2], 5) == [5, 0, 5]
    @test query_lengths([5, 2], 5, 5) === nothing
    @test query_lengths([5, 0, 1], 5, 3) == [5, 0, 3]
    @test query_lengths([3, 0, 1], 9, 3) == [5, 0, 3]
end

hidden(y::NamedTuple) = y.last_hidden_state
hidden(y) = y

# deterministic token indices of size (len, batch_size), as in kv_cache.jl
tokens(vocab, len, batch_size) = mod1.((1:len) .* 7 .+ (1:batch_size)' .* 13, vocab)

# The outputs of transformers' model (sdpa, float32) for the tokens above, in a batch of
# sequences of lengths n, 0 and 4: the first three features of the second sequence, which
# has no token, at its first and last position.
const ALL_PADDING = [
    (fixture = "qwen3_padded", vocab = 64, n = 9, # the attention of Llama, Qwen and Granite
        first = [-2.52641, 2.31819, -0.548628], last = [1.72846, -1.52322, -0.824479]),
    (fixture = "granite_options", vocab = 64, n = 9, # with a bias on the attention output
        first = [-0.0352943, -0.0895733, -0.0725985],
        last = [0.311012, 0.208099, 0.234994]),
    (fixture = "gemma3_options", vocab = 64, n = 9, # with sliding windows
        first = [1.69044, -0.422823, 1.23595], last = [0.902405, -0.385559, 1.20115]),
    (fixture = "bert_tiny", vocab = 64, n = 9,
        first = [-0.247188, 0.83853, -0.395303], last = [0.892097, 0.18227, -0.219495]),
    (fixture = "modernbert_tiny", vocab = 64, n = 9,
        first = [1.8297, -0.0527813, -0.463803], last = [0.917736, -1.3966, 0.322125]),
    (fixture = "clip_text", vocab = 64, n = 9,
        first = [1.21068, -0.396348, -0.426189], last = [0.449976, 1.70442, 0.167801]),
    (fixture = "qwen3_5_text_padded", vocab = 32, n = 12, # with linear attention
        first = [-1.67841, 2.74647, -2.90127], last = [3.11399, -3.44605, 3.14689]),
    (fixture = "siglip_tiny", vocab = 64, n = 8,
        first = [0.234779, 0.67904, 2.26798], last = [-0.21351, 1.41597, 1.41112])]

@testset "a sequence of all padding: $(case.fixture)" for case in ALL_PADDING
    model = fixture_model(fixture(case.fixture); eltype = Float32)
    model = hasproperty(model, :text_model) ? model.text_model : model
    ids = tokens(case.vocab, case.n, 3)
    mask = [i <= l for i in 1:(case.n), l in (case.n, 0, 4)]
    y = hidden(model(ids; attention_mask = mask))
    @test all(isfinite, y)
    @test isapprox(y[1:3, 1, 2], case.first; rtol = 1e-4, atol = 1e-4)
    @test isapprox(y[1:3, end, 2], case.last; rtol = 1e-4, atol = 1e-4)

    # the other sequences don't depend on it
    others = [1, 3]
    z = hidden(model(ids[:, others]; attention_mask = mask[:, others]))
    @test y[:, :, others] ≈ z

    # nor does a batch of it alone
    alone = hidden(model(ids[:, 2:2]; attention_mask = mask[:, 2:2]))
    @test alone ≈ y[:, :, 2:2]
end
