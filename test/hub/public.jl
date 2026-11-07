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

    # What the README, the tutorials and the API reference use or document.
    const USED = Dict(
        Metalhead => [:create_model, :create_processor, :list_models, :forward_features,
            :forward_head, :forward_intermediates, :feature_info,
            :post_process_object_detection, :post_process_instance_segmentation,
            :post_process_semantic_segmentation, :encode_query, :encode_document,
            :similarity, :Timm, :Transformers, :Diffusers, :SentenceTransformers,
            :TextProcessor, :image_array, :image_matrix],
        Timm => [:EvalTransform, :pretrained_cfg, :resnet, :vit],
        Transformers => [:KVCache, :last_tokens, :get_text_features, :get_image_features,
            :CLIPModel, :CLIPProcessor, :SiglipModel, :SiglipProcessor, :Siglip2Model,
            :Siglip2Processor, :Sam3Model, :Sam3Processor, :Qwen3VLProcessor,
            :Qwen3_5ForConditionalGeneration, :bert_model, :llama_for_causal_lm],
        Diffusers => [:Flux2KleinPipeline, :ZImagePipeline, :encode_prompt, :encode,
            :decode, :mode, :sample, :FlowMatchEulerDiscreteScheduler, :set_timesteps,
            :step],
        SentenceTransformers => [:SentenceTransformer, :encode, :encode_query,
            :encode_document, :similarity, :preprocess, :Transformer, :Pooling, :Dense,
            :Normalize])

    @testset "the documented API is public" begin
        for (m, names) in USED, name in names
            @test Base.ispublic(m, name)
        end
        # exported names are public too, the others are public only by declaration
        @test Base.isexported(Metalhead, :create_model)
        @test !Base.isexported(Timm, :pretrained_cfg)
    end

    @testset "the porting API is public" begin
        for name in (:ArchSpec, :register!, :AbstractConfigFormat, :architecture_of,
            :model_args, :skip_prefix, :weights_name, :instantiate, :Checkpoint,
            :load_state!)
            @test Base.ispublic(Metalhead, name)
            @test Base.Docs.hasdoc(Metalhead, name)
        end
    end

    @testset "internals are not public" begin
        for name in (:REGISTRY, :lookup, :direct_weight_init)
            @test !Base.ispublic(Metalhead, name)
        end
    end
else
    @testset "public API" begin
        @test_skip Base.ispublic(Metalhead, :Timm)
    end
end
