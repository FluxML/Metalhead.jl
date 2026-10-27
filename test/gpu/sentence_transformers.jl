using Metalhead, Test
include("../gpu_setup.jl")
using .GPUSetup

@testset "fixtures" begin
    names = fixtures_of(:SentenceTransformers)
    @test !isempty(names)
    @testset "$name" for name in names
        for r in check_gpu_st_fixture(name)
            @test r.on_gpu
            @test r.sentence_embedding <= GPU_RTOL
            @test r.token_embeddings <= GPU_RTOL
        end
    end
end
