using Metalhead, Test
using Metalhead: Timm, Transformers, Diffusers, SentenceTransformers

# `public` and `Base.ispublic` exist from Julia 1.11 on.
@static if VERSION >= v"1.11.0-DEV.469"
    const MODULES = [Metalhead, Timm, Transformers, Diffusers, SentenceTransformers]

    @testset "public and exported names have docstrings" begin
        for m in MODULES
            @test Base.Docs.undocumented_names(m) == Symbol[]
        end
    end

    # What the README, the tutorials and the API reference use or document, outside the
    # internals.
    const USED = Dict(
        Metalhead => [:create_model, :create_processor, :list_models, :forward_features,
            :forward_head, :forward_intermediates, :feature_info,
            :post_process_object_detection, :post_process_instance_segmentation,
            :post_process_semantic_segmentation, :encode_query, :encode_document,
            :similarity, :Timm, :Transformers, :Diffusers, :SentenceTransformers,
            :TextProcessor, :image_array, :image_matrix],
        Timm => [:EvalTransform, :pretrained_cfg],
        Transformers => [:KVCache, :last_tokens, :get_text_features, :get_image_features,
            :CLIPModel, :CLIPProcessor, :SiglipModel, :SiglipProcessor, :Siglip2Model,
            :Siglip2Processor, :Sam3Model, :Sam3Processor, :Qwen3VLProcessor,
            :Qwen3_5ForConditionalGeneration, :BertModel, :TextEncoder],
        Diffusers => [:Flux2KleinPipeline, :ZImagePipeline, :encode_prompt],
        SentenceTransformers => [:SentenceTransformer, :encode, :encode_query,
            :encode_document, :similarity])

    @testset "the documented API is public" begin
        for (m, names) in USED, name in names
            @test Base.ispublic(m, name)
        end
        # exported names are public too, the others are public only by declaration
        @test Base.isexported(Metalhead, :create_model)
        @test !Base.isexported(Timm, :pretrained_cfg)
    end

    # What mirrors upstream's internals or keyword arguments, or only serves the porting.
    @testset "internals are not public" begin
        INTERNAL = Dict(
            Metalhead => [:ArchSpec, :register!, :AbstractConfigFormat, :architecture_of,
                :model_args, :skip_prefix, :weights_name, :instantiate, :Checkpoint,
                :load_state!, :checkpoint_prefix, :AbstractProcessor, :REGISTRY, :lookup],
            Timm => [:resnet, :vit, :efficientnet, :pil_resize, :create_conv2d],
            Transformers => [:bert_model, :llama_for_causal_lm, :clip_model, :sam3_model,
                :get_vision_features, :qwen3_5_decode, :MultiheadAttention],
            Diffusers => [:encode, :decode, :mode, :sample, :step, :set_timesteps,
                :FlowMatchEulerDiscreteScheduler, :DiagonalGaussianDistribution,
                :Flux2Transformer2DModel, :autoencoder_kl, :flux2_transformer_2d_model],
            SentenceTransformers => [:preprocess, :Transformer, :Pooling, :Dense,
                :Normalize])
        for (m, names) in INTERNAL, name in names
            @test !Base.ispublic(m, name)
        end
    end
else
    @testset "public API" begin
        @test_skip Base.ispublic(Metalhead, :Timm)
    end
end
