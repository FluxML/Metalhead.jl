using Metalhead, Test
include("../gpu_setup.jl")
using .GPUSetup
using Flux: Flux, BFloat16
using CUDA: CuArray

const D = Metalhead.Diffusers

@testset "fixtures" begin
    @testset "$name" for name in filter(!is_pipeline, fixtures_of(:Diffusers))
        r = check_gpu_fixture(name)
        @test r.on_gpu
        @test r.output <= GPU_RTOL
    end
end

@testset "pipelines" begin
    @testset "$name" for name in filter(is_pipeline, fixtures_of(:Diffusers))
        r = check_pipeline_fixture(fixture(name); eltype = Float32, device = GPU)
        for err in r
            @test err <= GPU_RTOL
        end
    end
end

@testset "tokenizer" begin
    # `gpu` and `bf16` leave the tokenizer of a pipeline on the CPU, as it is
    p = create_model(fixture("flux2_klein_tiny"))
    tok = Metalhead.Transformers.TokenizersBackend(; pad_token = "<pad>")
    pipe = D.Flux2KleinPipeline(p.scheduler, p.vae, p.text_encoder, tok, p.transformer,
        p.is_distilled)
    @test GPU(pipe).tokenizer === tok
    @test Flux.bf16(GPU(pipe)).tokenizer === tok
end

@testset "gradient" begin
    # vae_flux2_options has no attention in its mid blocks
    for name in ["vae_flux2_options", "transformer_flux2_tiny"]
        model = GPU(fixture_model(fixture(name); eltype = Float32))
        @test check_gradient(model, first(gpu_inputs(name))...)
    end
end

@testset "bfloat16" begin
    model = GPU(fixture_model(fixture("vae_flux2_options"); eltype = Float32))
    @test check_half_precision(model, first(gpu_inputs("vae_flux2_options"))...;
        T = BFloat16, rtol = BF16_RTOL)
    pipe = GPU(create_model(fixture("flux2_klein_tiny"); pretrained = true,
        eltype = BFloat16))
    embeds = D.encode_prompt(pipe, rand(1:32, 7, 2); text_encoder_out_layers = (1, 2, 3))
    @test embeds isa CuArray{BFloat16}
    @test_logs min_level=Base.CoreLogging.Warn begin
        image = pipe(embeds; height = 16, width = 20, num_inference_steps = 2)
        @test image isa CuArray{BFloat16}
    end
end
