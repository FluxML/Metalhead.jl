using Metalhead, Test
using Flux: Flux
include("../parity_setup.jl")
using .ParitySetup

const D = Metalhead.Diffusers
const FLUX2_FIXTURES = ["vae_flux2_tiny", "vae_flux2_options", "transformer_flux2_tiny",
    "transformer_flux2_options"]
const FLUX2_PIPELINES = ["flux2_klein_tiny", "flux2_klein_cfg"]

f32(x) = x isa AbstractArray{Float64} ? Float32.(x) : x
fixture_args(name) = map(f32, first(fixture_inputs(Metalhead.Checkpoint(
    joinpath(fixture(name), "acts.safetensors")))))

@testset "fixtures" begin
    for name in FLUX2_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.pass
    end
end

@testset "pipelines" begin
    for name in FLUX2_PIPELINES
        r = check_pipeline_fixture(fixture(name))
        @test r.prompt_embeds <= 1e-8
        @test r.negative_prompt_embeds <= 1e-8
        # diffusers keeps the noise levels, and the eps of the statistics of the latents, in
        # Float32 even in a Float64 pipeline
        @test r.latents <= 1e-10
        @test r.image <= 1e-10
    end
end

@testset "gradient" begin
    for name in FLUX2_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        @test check_gradient(model, fixture_args(name)...)
    end
end

@testset "half precision" begin
    # As diffusers does, the transformer scales the timesteps and the guidance by 1000 in the
    # eltype of the model: the guidance scales of up to 5 of the options fixture lose 7% of
    # their time embedding in 16 bits. The other fixtures have reached 1.1e-2 in CI.
    for name in FLUX2_FIXTURES
        rtol = name == "transformer_flux2_options" ? 0.1 : 2e-2
        @test check_half_precision(name; rtol)
    end
    pipe = create_model(fixture("flux2_klein_tiny"); pretrained = true, eltype = Float16)
    @test eltype(pipe.vae.bn.μ) == eltype(pipe.text_encoder.model.norm.weight) == Float32
    embeds = D.encode_prompt(pipe, rand(1:32, 7, 2); text_encoder_out_layers = (1, 2, 3))
    @test eltype(embeds) == Float16
    @test_logs min_level=Base.CoreLogging.Warn begin
        image = pipe(embeds; height = 16, width = 20, num_inference_steps = 2)
        @test eltype(image) == Float16
    end
end

@testset "autoencoder" begin
    tiny = (; block_out_channels = (8, 16), layers_per_block = 1, latent_channels = 4,
        norm_num_groups = 4, down_block_types = ntuple(_ -> "DownEncoderBlock2D", 2),
        up_block_types = ntuple(_ -> "UpDecoderBlock2D", 2))
    vae = create_model("AutoencoderKLFlux2"; tiny...)
    x = randn(Float32, 16, 12, 3, 2)
    posterior = D.encode(vae, x)
    @test size(D.mode(posterior)) == (8, 6, 4, 2)
    @test size(D.sample(posterior)) == (8, 6, 4, 2)
    @test D.sample(posterior) != D.sample(posterior)
    @test all(-30 .<= posterior.logvar .<= 20)
    @test size(vae(x)) == size(x)
    @test size(vae.bn.μ) == (16,)
    @test_throws "DownEncoderBlock2D" create_model("AutoencoderKLFlux2"; tiny...,
        down_block_types = ("DownEncoderBlock2D", "AttnDownEncoderBlock2D"))
    @test_throws "same length" create_model("AutoencoderKLFlux2"; tiny...,
        block_out_channels = (8, 16, 16))
    @test_throws "not supported" create_model("AutoencoderKLFlux2"; tiny..., act_fn = "gelu")
end

@testset "transformer" begin
    tiny = (; num_layers = 1, num_single_layers = 1, attention_head_dim = 8,
        num_attention_heads = 2, joint_attention_dim = 6, in_channels = 4,
        axes_dims_rope = (2, 2, 2, 2), timestep_guidance_channels = 8)
    model = create_model("Flux2Transformer2DModel"; tiny..., guidance_embeds = false)
    x, context = randn(Float32, 4, 6, 2), randn(Float32, 6, 3, 2)
    y = model(x, context, Float32[0.5, 0.1], D.latent_ids(3, 2), D.text_ids(3))
    @test size(y) == (4, 6, 2)
    # diffusers takes the position ids of the first batch element
    ids = cat(D.latent_ids(3, 2), zeros(Int, 4, 6); dims = 3)
    @test model(x, context, Float32[0.5, 0.1], ids, D.text_ids(3)) ≈ y
    @test model.time_guidance_embed.guidance_embedder === nothing
    guided = create_model("Flux2Transformer2DModel"; tiny..., out_channels = 2, patch_size = 2)
    @test size(guided(x, context, Float32[0.5, 0.1], D.latent_ids(3, 2), D.text_ids(3),
        Float32[3, 4])) == (8, 6, 2)
    @test_throws "axes_dims_rope" create_model("Flux2Transformer2DModel"; tiny...,
        axes_dims_rope = (2, 2, 2))
end

@testset "scheduler" begin
    s = D.flow_match_euler_discrete_scheduler(; shift = 3.0)
    @test last(D.set_timesteps(s, 5)) ≈
          [1.0, 0.9003590941429138, 0.7511211037635803, 0.5029850602149963,
        0.008928571827709675, 0.0]
    s = D.flow_match_euler_discrete_scheduler(; shift = 2.0)
    timesteps, sigmas = D.set_timesteps(s, 4; sigmas = [1.0, 0.75, 0.5, 0.25])
    @test sigmas ≈ [1.0, 0.8571428656578064, 0.6666666865348816, 0.4000000059604645, 0.0]
    @test timesteps ≈ [1000.0, 857.1428833007812, 666.6666870117188, 400.0]
    @test eltype(timesteps) == eltype(sigmas) == Float32
    linear = D.flow_match_euler_discrete_scheduler(; use_dynamic_shifting = true,
        time_shift_type = "linear")
    @test last(D.set_timesteps(linear, 3; mu = 1.5)) ≈
          [1.0, 0.6004799008369446, 0.0014992504147812724, 0.0]
    exponential = D.flow_match_euler_discrete_scheduler(; use_dynamic_shifting = true)
    @test last(D.set_timesteps(exponential, 3; mu = 0.8)) ≈
          [1.0, 0.6904021501541138, 0.002222816925495863, 0.0]
    @test_throws "mu" D.set_timesteps(exponential, 3)
    @test_throws "stochastic_sampling" D.flow_match_euler_discrete_scheduler(;
        stochastic_sampling = true)
    @test D.compute_empirical_mu(256, 4) ≈ 1.965567161408539
    @test D.compute_empirical_mu(5000, 4) ≈ 1.30301666
    sample = ones(Float16, 2)
    @test D.step(s, Float16[1, 2], 0.5, 0.25, sample) == Float16[0.75, 0.5]
end

@testset "pipeline" begin
    @test list_models(; byfamily = true)["flux2"] ==
          ["AutoencoderKLFlux2", "Flux2KleinPipeline", "Flux2Transformer2DModel"]
    @test_throws "built from a repository" create_model("Flux2KleinPipeline")
    dir = fixture("flux2_klein_tiny")
    pipe = create_model(dir)
    @test pipe isa D.Flux2KleinPipeline
    @test pipe.is_distilled && pipe.tokenizer === nothing
    @test pipe.text_encoder isa Metalhead.Transformers.LlamaForCausalLM
    @test create_model(dir; subfolder = "vae") isa D.AutoencoderKLFlux2
    @test_throws "has no tokenizer to encode prompts" pipe("a prompt")
    @test_throws "has no tokenizer to encode prompts" D.encode_prompt(pipe, "a prompt")
    embeds = randn(Float32, 48, 7, 1)
    image = pipe(embeds; height = 16, width = 20, num_inference_steps = 2)
    @test size(image) == (20, 16, 3, 1) && eltype(image) == Float32
    @test all(0 .<= image .<= 1)
    latents = pipe(embeds; height = 16, width = 20, num_inference_steps = 1,
        output_type = :latent)
    @test size(latents) == (10, 8, 4, 1)
    @test_throws "latents" pipe(embeds; height = 16, width = 20, latents = zeros(4, 4, 16, 1))
    base = create_model(fixture("flux2_klein_cfg"))
    @test !base.is_distilled
    @test_throws "negative_prompt_embeds" base(embeds; height = 16, width = 20)
    @test size(base(embeds; height = 16, width = 20, num_inference_steps = 1,
        negative_prompt_embeds = randn(Float32, 48, 3, 1))) == (20, 16, 3, 1)
    @test D.patchify_latents(D.unpatchify_latents(latents)) == latents
end

# Tokenizes the chat prompts of a pipeline into the small vocabulary of the fixtures, with
# the padding token 0.
struct PromptTokenizer end
function Metalhead.encode_ids(::PromptTokenizer, text::AbstractString;
        add_special_tokens::Bool = true)
    return [Int(c) % 30 + 1 for c in text]
end
Metalhead.pad_token_id(::PromptTokenizer) = 0

with_tokenizer(p, tokenizer) = D.Flux2KleinPipeline(p.scheduler, p.vae, p.text_encoder,
    tokenizer, p.transformer, p.is_distilled)

@testset "tokenizer" begin
    # the tokenizer of a pipeline is HuggingFaceTokenizers' of its tokenizer/ subfolder,
    # which a pipeline loaded without HuggingFaceTokenizers loads once it is (with
    # METALHEAD_TEST_NETWORK, another test of this worker may have loaded it)
    if Base.get_extension(Metalhead, :MetalheadHuggingFaceTokenizersExt) === nothing
        mktempdir() do dir
            cp(fixture("flux2_klein_tiny"), dir; force = true)
            mkdir(joinpath(dir, "tokenizer"))
            write(joinpath(dir, "tokenizer", "tokenizer.json"), "{}")
            index = JSON.parsefile(joinpath(dir, "model_index.json"))
            index["tokenizer"] = ["transformers", "Qwen2TokenizerFast"]
            write(joinpath(dir, "model_index.json"), JSON.json(index))
            pipe = create_model(dir)
            @test pipe.tokenizer ==
                  Metalhead.DeferredTokenizer(joinpath(dir, "tokenizer", "tokenizer.json"),
                nothing)
            @test Flux.f16(pipe).tokenizer === pipe.tokenizer
            @test_throws "run `using HuggingFaceTokenizers`, and call the pipeline again" pipe(
                "a prompt")
        end
    end

    # the prompts tokenized, padded and truncated to `max_sequence_length` as diffusers
    # does: the chat prompt of the first has fewer than 80 characters, that of the second
    # more
    pipe = with_tokenizer(create_model(fixture("flux2_klein_tiny"); pretrained = true,
            eltype = Float32), PromptTokenizer())
    prompts = ["a cat", "two dogs on a sofa, in the sun"]
    codes = [Metalhead.encode_ids(PromptTokenizer(), D.chat_prompt(p)) for p in prompts]
    n = length(codes[1])
    @test n < 80 < length(codes[2])
    ids = [[codes[1]; zeros(Int, 80 - n)] codes[2][1:80]] .+ 1
    mask = [[ones(Int, n); zeros(Int, 80 - n)] ones(Int, 80)]
    @test D.qwen3_prompt_inputs(pipe.tokenizer, prompts; max_sequence_length = 80) ==
          (; input_ids = ids, attention_mask = mask)
    encode(p, prompt) = D.encode_prompt(p, prompt; max_sequence_length = 80,
        text_encoder_out_layers = (1, 2, 3))
    embeds = encode(pipe, prompts)
    @test size(embeds) == (48, 80, 2)
    @test embeds ≈ D.encode_prompt(pipe, ids; attention_mask = mask,
        text_encoder_out_layers = (1, 2, 3))
    @test encode(pipe, "a cat") ≈ embeds[:, :, 1:1]
    # the string call of the pipeline goes through `encode_prompt`
    kwargs = (; height = 16, width = 20, num_inference_steps = 2,
        latents = randn(Float32, 5, 4, 16, 1))
    text = (; max_sequence_length = 80, text_encoder_out_layers = (1, 2, 3))
    @test pipe("a cat"; kwargs..., text...) ≈ pipe(embeds[:, :, 1:1]; kwargs...)
    # classifier-free guidance, with the empty prompt as negative prompt
    base = with_tokenizer(create_model(fixture("flux2_klein_cfg"); pretrained = true,
            eltype = Float32), PromptTokenizer())
    @test base("a cat"; kwargs..., text...) ≈ base(encode(base, "a cat");
        negative_prompt_embeds = encode(base, ""), kwargs...)

    # diffusers components have no processor; the tokenizer subfolder is a tokenizer
    dir = fixture("flux2_klein_tiny")
    @test_throws "pipe.tokenizer" create_processor(dir)
    component = "AutoencoderKLFlux2, a component of a diffusers pipeline, has no processor"
    @test_throws component create_processor(dir; subfolder = "vae")
    @test_throws "has no processor" create_processor(dir; subfolder = "transformer")
end

TEST_NETWORK && @testset "references" begin
    # the transformer and the pipeline download 8 and 16 GB
    archs = TEST_LARGE ? ["AutoencoderKLFlux2", "Flux2Transformer2DModel", "Flux2KleinPipeline"] :
            ["AutoencoderKLFlux2"]
    for arch in archs, file in reference_files_of(arch)
        r = check_diffusers_reference(file)
        for (name, err) in pairs(r)
            name == :repo || @test err <= 1e-3
        end
    end
end
