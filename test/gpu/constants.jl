using Metalhead, Test
include("../gpu_setup.jl")
using .GPUSetup
using CUDA: CuArray
using Flux: BFloat16

const T = Metalhead.Transformers
const D = Metalhead.Diffusers

@testset "rotary tables" begin
    params = [(; rope_type = "default", rope_theta = 10_000.0),
        (; rope_type = "linear", rope_theta = 500_000.0, factor = 8.0),
        (; rope_type = "llama3", rope_theta = 500_000.0, factor = 8.0,
            low_freq_factor = 1.0, high_freq_factor = 4.0,
            original_max_position_embeddings = 8192)]
    for E in (Float16, BFloat16, Float32, Float64), n in (7, 131), p in params
        x = zeros(E, 16, n, 2)
        m = T.RotaryEmbedding(16, p)
        expected = T.rotary_tables(m, x, n)
        actual = T.rotary_tables(m, CuArray(x), n)
        @test all(a -> a isa CuArray{E}, actual)
        @test all(Array(a) ≈ e for (a, e) in zip(actual, expected))
    end
end

@testset "multimodal rotary tables" begin
    m = T.Qwen3_5TextRotaryEmbedding(16, 10_000.0, (2, 1, 2))
    for E in (Float16, BFloat16, Float32, Float64), n in (7, 131), batch in (1, 2)
        x = zeros(E, 16, n, 2)
        positions = reshape(collect(1:(3n * batch)), 3, n, batch)
        for ids in (nothing, positions, GPU(positions), cat(positions[1:1, :, :], positions; dims = 1))
            expected = T.rotary_tables(m, x, ids === nothing ? nothing : Array(ids))
            actual = T.rotary_tables(m, CuArray(x), ids)
            @test all(a -> a isa CuArray{E}, actual)
            @test all(Array(a) ≈ e for (a, e) in zip(actual, expected))
        end
    end
end

@testset "FLUX.2 rotary tables" begin
    m = D.Flux2PosEmbed(2000.0, (2, 4, 6, 8))
    for E in (Float16, BFloat16, Float32, Float64), fractional in (false, true)
        ids = reshape(collect(-22:21), 4, 11)
        ids = fractional ? ids ./ 3 : ids
        x = zeros(E, 20, 11, 2)
        expected = D.rotary_tables(m, ids, x)
        for positions in (ids, CuArray(ids))
            actual = D.rotary_tables(m, positions, CuArray(x))
            @test all(a -> a isa CuArray{E}, actual)
            @test all(Array(a) ≈ e for (a, e) in zip(actual, expected))
        end
    end
end

@testset "chunk masks" begin
    for E in (Float16, BFloat16, Float32, Float64), C in (8, 64)
        x = CuArray(zeros(E, 1))
        causal = T.causal_log_mask(x, C)
        @test causal isa CuArray{E}
        @test Array(causal) == [j <= i ? zero(E) : E(-Inf) for j in 1:C, i in 1:C]
        identity = T.identity_matrix(x, C)
        @test identity isa CuArray{E}
        @test Array(identity) == [E(i == j) for i in 1:C, j in 1:C]
        for s in (1, 2, 4)
            mask = T.block_mask(x, C, s)
            @test mask isa CuArray{E}
            expected = [E((i - 1) ÷ 2s == (j - 1) ÷ 2s &&
                              iseven((i - 1) ÷ s) && isodd((j - 1) ÷ s))
                        for i in 1:C, j in 1:C]
            @test Array(mask) == expected
        end
    end
end
