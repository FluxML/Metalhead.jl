using Metalhead, Test
using Flux: Flux
include("../parity_setup.jl")
using .ParitySetup

const D = Metalhead.Diffusers
const Z_IMAGE_FIXTURES = ["autoencoder_kl_tiny", "autoencoder_kl_options",
    "transformer_z_image_tiny", "transformer_z_image_options"]
const Z_IMAGE_PIPELINES = ["z_image_tiny", "z_image_cfg"]

f32(x::AbstractArray{Float64}) = Float32.(x)
f32(x::AbstractVector{<:AbstractArray}) = map(f32, x)
f32(x) = x
fixture_acts(name) = Metalhead.Checkpoint(joinpath(fixture(name), "acts.safetensors"))
fixture_args(name) = map(f32, first(fixture_inputs(fixture_acts(name))))
function fixture_meta(name)
    return SafeTensors.deserialize(joinpath(fixture(name), "acts.safetensors")).metadata
end

# The options fixture has the patch embedding of a second pair of patch sizes, which the
# default `patch_size` and `f_patch_size` don't use.
function unused_modules(name, model)
    name == "transformer_z_image_options" || return ()
    return (model.all_x_embedder[Symbol("1-2")], model.all_final_layer[Symbol("1-2")])
end

@testset "fixtures" begin
    for name in Z_IMAGE_FIXTURES
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.pass
    end
end

@testset "parameter counts" begin
    for name in Z_IMAGE_FIXTURES
        model = fixture_model(fixture(name))
        @test sum(length, Flux.trainables(model)) ==
              parse(Int, fixture_meta(name)["num_params"])
    end
end

@testset "pipelines" begin
    for name in Z_IMAGE_PIPELINES
        r = check_pipeline_fixture(fixture(name))
        @test r.prompt_embeds <= 1e-8
        @test r.negative_prompt_embeds <= 1e-8
        @test r.latents <= 1e-10
        @test r.image <= 1e-10
    end
end

@testset "gradient" begin
    for name in Z_IMAGE_FIXTURES
        model = fixture_model(fixture(name); eltype = Float32)
        unused = unused_modules(name, model)
        @test check_gradient(model, fixture_args(name)...; unused)
    end
end

@testset "half precision" begin
    # As diffusers does, the transformer scales the timesteps by `t_scale` in their eltype,
    # here Float16
    for name in Z_IMAGE_FIXTURES
        rtol = startswith(name, "transformer") ? 5e-2 : 2e-2
        @test check_half_precision(name; rtol)
    end
    pipe = create_model(fixture("z_image_tiny"); pretrained = true, eltype = Float16)
    @test eltype(pipe.text_encoder.norm.weight) == Float32
    @test eltype(pipe.vae.encoder.conv_norm_out.γ) == Float32
    embeds = D.encode_prompt(pipe, rand(1:32, 7, 2))
    @test eltype(first(embeds)) == Float16
    logged, _ = warnings() do
        image = pipe(embeds; height = 12, width = 16, num_inference_steps = 2,
            guidance_scale = 0)
        @test eltype(image) == Float16
        # the latents are denoised in Float32, as diffusers does
        latents = pipe(embeds; height = 12, width = 16, num_inference_steps = 2,
            guidance_scale = 0, output_type = :latent)
        @test eltype(latents) == Float32
    end
    @test isempty(logged)
end

@testset "autoencoder" begin
    tiny = (; block_out_channels = (8, 16), layers_per_block = 1, latent_channels = 4,
        norm_num_groups = 4, down_block_types = ntuple(_ -> "DownEncoderBlock2D", 2),
        up_block_types = ntuple(_ -> "UpDecoderBlock2D", 2))
    vae = create_model("AutoencoderKL"; tiny...)
    x = randn(Float32, 16, 12, 3, 2)
    posterior = D.encode(vae, x)
    @test size(D.mode(posterior)) == (8, 6, 4, 2)
    @test size(vae(x)) == size(x)
    @test vae.scaling_factor == 0.18215 && vae.shift_factor === nothing
    @test size(vae.quant_conv.weight) == (1, 1, 8, 8)
    plain = create_model("AutoencoderKL"; tiny..., use_quant_conv = false,
        use_post_quant_conv = false, shift_factor = 0.1159)
    @test plain.quant_conv === plain.post_quant_conv === nothing
    @test plain.shift_factor == 0.1159
    @test_throws "DownEncoderBlock2D" create_model("AutoencoderKL"; tiny...,
        down_block_types = ("DownEncoderBlock2D", "AttnDownEncoderBlock2D"))
end

@testset "transformer" begin
    tiny = (; dim = 32, n_layers = 1, n_refiner_layers = 1, n_heads = 2, cap_feat_dim = 6,
        in_channels = 4, axes_dims = (4, 6, 6), axes_lens = (80, 8, 8))
    model = create_model("ZImageTransformer2DModel"; tiny...)
    x = randn(Float32, 6, 4, 4, 2)
    t = Float32[0.5, 0.1]
    caps = randn(Float32, 6, 5, 2)
    y = model(x, t, caps)
    @test size(y) == size(x)
    # a batch of images of one frame, and captions of the same length as an array
    @test model(reshape(x, 6, 4, 1, 4, 2), t, caps) ≈ reshape(y, 6, 4, 1, 4, 2)
    @test model(x, t, [caps[:, :, 1], caps[:, :, 2]]) ≈ y
    # each image depends on its caption only, padded to 32 tokens or 64
    long = [caps[:, :, 1], randn(Float32, 6, 40)]
    @test model(x, t, long)[:, :, :, 1] ≈ y[:, :, :, 1]
    @test model(x[:, :, :, 1:1], t[1:1], long[1:1]) ≈ y[:, :, :, 1:1]
    @test model.in_channels == 4 && model.t_scale == 1000
    @test_throws "patch embedding" model(x, t, caps; patch_size = 1)
    @test_throws "3 captions" model(x, t, [caps[:, :, 1], caps[:, :, 2], caps[:, :, 1]])
    @test_throws "divide into patches" model(x[1:5, :, :, :], t, caps)
    # the frames of the captions run past `axes_lens`
    @test_throws "axes_lens" model(x, t, [randn(Float32, 6, 70), caps[:, :, 2]])
    frames = create_model("ZImageTransformer2DModel"; tiny..., all_patch_size = (2, 1),
        all_f_patch_size = (1, 2))
    video = randn(Float32, 6, 4, 2, 4, 2)
    @test size(frames(video, t, caps; patch_size = 1, f_patch_size = 2)) == size(video)
    @test_throws "Omni" create_model("ZImageTransformer2DModel"; tiny...,
        siglip_feat_dim = 8)
    @test_throws "head dimension" create_model("ZImageTransformer2DModel"; tiny...,
        axes_dims = (4, 4, 4))
    @test_throws "same length" create_model("ZImageTransformer2DModel"; tiny...,
        all_patch_size = (2, 1))
end

@testset "sampling" begin
    # torch.linspace(1, 1 / 9, 9) in float32, of Z-Image-Turbo's 9 steps
    @test D.z_image_sigmas(9) ==
          Float32[1.0, 0.8888888955116272, 0.7777777910232544, 0.6666666865348816,
        0.5555555820465088, 0.4444444477558136, 0.3333333432674408, 0.2222222238779068,
        0.1111111119389534]
    @test D.z_image_sigmas(1) == [1.0f0]
    pos, neg = ones(Float32, 2, 2, 1, 2), zeros(Float32, 2, 2, 1, 2)
    neg[:, :, :, 2] .= 0.9
    guided = D.guided_flow(pos, neg, 2.0f0, 0.0f0)
    @test guided[:, :, :, 1] == fill(3, 2, 2, 1) && guided[:, :, :, 2] ≈ fill(1.2, 2, 2, 1)
    # normalized to at most 1.5 times the norm of `pos`
    normalized = D.guided_flow(pos, neg, 2.0f0, 1.5f0)
    @test normalized[:, :, :, 1] ≈ fill(1.5, 2, 2, 1)
    @test normalized[:, :, :, 2] == guided[:, :, :, 2]
end

@testset "pipeline" begin
    @test list_models(; byfamily = true)["z_image"] ==
          ["AutoencoderKL", "ZImagePipeline", "ZImageTransformer2DModel"]
    @test_throws "built from a repository" create_model("ZImagePipeline")
    dir = fixture("z_image_tiny")
    pipe = create_model(dir)
    @test pipe isa D.ZImagePipeline
    @test pipe.tokenizer === nothing
    # the pipeline holds the decoder of the text encoder's `Qwen3ForCausalLM` checkpoint, as
    # diffusers loads its `Qwen3Model`
    @test pipe.text_encoder isa Metalhead.Transformers.LlamaModel
    @test create_model(dir; subfolder = "text_encoder") isa
          Metalhead.Transformers.LlamaForCausalLM
    @test create_model(dir; subfolder = "vae") isa D.AutoencoderKL
    @test create_model(dir; subfolder = "transformer") isa D.ZImageTransformer2DModel
    @test_throws "has no tokenizer to encode prompts" pipe("a prompt")
    @test_throws "has no tokenizer to encode prompts" D.encode_prompt(pipe, "a prompt")
    mask = [ones(Int, 7) [1, 1, 1, 0, 0, 0, 0]]
    embeds = D.encode_prompt(pipe, rand(1:32, 7, 2); attention_mask = mask)
    @test size.(embeds) == [(16, 7), (16, 3)]
    image = pipe(embeds; height = 12, width = 16, num_inference_steps = 2,
        guidance_scale = 0)
    @test size(image) == (16, 12, 3, 2) && eltype(image) == Float32
    @test all(0 .<= image .<= 1)
    latents = pipe(embeds; height = 12, width = 16, num_inference_steps = 1,
        guidance_scale = 0, output_type = :latent)
    @test size(latents) == (8, 6, 4, 2)
    @test_throws "divisible by 4" pipe(embeds; height = 10, width = 16, guidance_scale = 0)
    @test_throws "latents" pipe(embeds; height = 12, width = 16, guidance_scale = 0,
        latents = zeros(4, 4, 4, 2))
    @test_throws "negative_prompt_embeds" pipe(embeds; height = 12, width = 16)
    @test size(pipe(embeds; height = 12, width = 16, num_inference_steps = 1,
        negative_prompt_embeds = [randn(Float32, 16, 2), randn(Float32, 16, 3)])) ==
          (16, 12, 3, 2)
end

# Tokenizes the chat prompts of a pipeline into the small vocabulary of the fixtures, with
# the padding token 0.
struct PromptTokenizer end
function Metalhead.encode_ids(::PromptTokenizer, text::AbstractString;
        add_special_tokens::Bool = true)
    return [Int(c) % 30 + 1 for c in text]
end
Metalhead.pad_token_id(::PromptTokenizer) = 0

with_tokenizer(p, tokenizer) = D.ZImagePipeline(p.scheduler, p.vae, p.text_encoder,
    tokenizer, p.transformer)

@testset "tokenizer" begin
    pipe = with_tokenizer(create_model(fixture("z_image_cfg"); pretrained = true,
            eltype = Float32), PromptTokenizer())
    # thinking enabled: no empty thinking block
    @test D.z_image_chat_prompt("a cat") ==
          "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
    prompts = ["a cat", "two dogs on a sofa, in the sun"]
    codes = [Metalhead.encode_ids(PromptTokenizer(), D.z_image_chat_prompt(p))
             for p in prompts]
    n = length(codes[1])
    @test n < 70 < length(codes[2])
    ids = [[codes[1]; zeros(Int, 70 - n)] codes[2][1:70]] .+ 1
    mask = [[ones(Int, n); zeros(Int, 70 - n)] ones(Int, 70)]
    @test D.z_image_prompt_inputs(pipe.tokenizer, prompts; max_sequence_length = 70) ==
          (; input_ids = ids, attention_mask = mask)
    embeds = D.encode_prompt(pipe, prompts; max_sequence_length = 70)
    @test size.(embeds) == [(16, n), (16, 70)]
    @test embeds ≈ D.encode_prompt(pipe, ids; attention_mask = mask)
    # the string call of the pipeline goes through `encode_prompt`, with the empty negative
    # prompt for classifier-free guidance
    kwargs = (; height = 16, width = 8, num_inference_steps = 2,
        latents = randn(Float32, 4, 8, 4, 1))
    @test pipe("a cat"; kwargs..., guidance_scale = 0) ≈
          pipe(embeds[1:1]; kwargs..., guidance_scale = 0)
    negative = D.encode_prompt(pipe, "")
    @test pipe("a cat"; kwargs...) ≈ pipe(embeds[1:1]; negative_prompt_embeds = negative,
        kwargs...)
    dog = D.encode_prompt(pipe, "a dog")
    @test pipe("a cat"; negative_prompt = "a dog", kwargs...) ≈
          pipe(embeds[1:1]; negative_prompt_embeds = dog, kwargs...)
    @test_throws "negative prompts" pipe(prompts; negative_prompt = "a dog", kwargs...)

    # diffusers components have no processor; the tokenizer subfolder is a tokenizer
    dir = fixture("z_image_tiny")
    @test_throws "pipe.tokenizer" create_processor(dir)
    component = "AutoencoderKL, a component of a diffusers pipeline, has no processor"
    @test_throws component create_processor(dir; subfolder = "vae")
end

TEST_NETWORK && @testset "references" begin
    # the transformer and the pipeline download 25 and 33 GB (in float32 on the CPU, they
    # take as much memory, the pipeline 42 GB)
    archs = TEST_LARGE ? ["AutoencoderKL", "ZImageTransformer2DModel", "ZImagePipeline"] :
            ["AutoencoderKL"]
    for arch in archs, file in reference_files_of(arch)
        r = check_diffusers_reference(file)
        for (name, err) in pairs(r)
            name == :repo || @test err <= 1e-3
        end
    end
end
