"""
    Metalhead.Transformers

Flux ports of [Hugging Face Transformers](https://github.com/huggingface/transformers) models.
Each ported model mirrors the PyTorch module tree, so that pretrained `state_dict`s load by
name; build them with [`Metalhead.create_model`](@ref) from a Hub repository.

Language models take token indices starting at 1, as Flux's `Embedding` does: add 1 to the
ids of a Hugging Face tokenizer, and subtract 1 from the index of a logit to get its token id.
With `using HuggingFaceTokenizers`, [`Metalhead.create_processor`](@ref) loads the tokenizer
of a repository.
"""
module Transformers

using Flux
using NNlib
using ConcreteStructs: @concrete
using ChainRulesCore: ChainRulesCore
using Functors: Functors
using MLUtils: topk
using ..Metalhead: ArchSpec, AbstractConfigFormat, register!, declared_args
using ..Metalhead: init_layer, init_array
using ..Metalhead: AbstractProcessor, AbstractImageProcessor, register_processor!,
                   image_batch, convert_to_rgb, is_8bit, RESAMPLE_FILTERS, check_inputs,
                   @processor, tokenizer_of, encode_ids, token_id, pad_token_id,
                   pad_ids, text_list
using ..Timm: dropout_or_identity, attention_fdrop, feature_take_indices, stage_outputs,
              pil_resize
import ..Metalhead: layer_params, architecture_of, model_args, direct_weight_init,
                    checkpoint_prefix
import ..Metalhead: forward_features, forward_head, forward_intermediates, feature_info
import ..Metalhead: post_process_object_detection, post_process_instance_segmentation,
                    post_process_semantic_segmentation
import ..Metalhead: ignored_keys, legacy_keys, standardize_settings

include("config.jl")
include("layers.jl")
include("llama.jl")
include("qwen2.jl")
include("qwen3.jl")
include("qwen3_moe.jl")
include("gemma3.jl")
include("granite.jl")
include("granitemoe.jl")
include("qwen3_5.jl")
include("qwen3_5_vision.jl")
include("qwen3_5_moe.jl")
include("image_processing_backends.jl")
include("image_processing_qwen2_vl.jl")
include("processing_qwen3_vl.jl")
include("clip.jl")
include("image_processing_clip.jl")
include("processing_clip.jl")
include("sam3.jl")
include("image_processing_sam3.jl")
include("processing_sam3.jl")
include("siglip.jl")
include("siglip2.jl")
include("image_processing_siglip.jl")
include("image_processing_siglip2.jl")
include("processing_siglip.jl")
include("processing_siglip2.jl")
include("bert.jl")
include("xlm_roberta.jl")
include("modernbert.jl")

end # module
