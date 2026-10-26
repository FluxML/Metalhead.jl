"""
    Metalhead.Transformers

Flux ports of [Hugging Face Transformers](https://github.com/huggingface/transformers) models.
Each ported model mirrors the PyTorch module tree, so that pretrained `state_dict`s load by
name; build them with [`Metalhead.create_model`](@ref) from a Hub repository.

Language models take token indices starting at 1, as Flux's `Embedding` does: add 1 to the
ids of a Hugging Face tokenizer, and subtract 1 from the index of a logit to get its token id.
The tokenizer of a repository, `create_processor(repo)`, gives them, and
[`decode`](@ref Metalhead.Transformers.decode) takes them.
"""
module Transformers

using Flux
using NNlib
using ConcreteStructs: @concrete
using ChainRulesCore: ChainRulesCore
using MLUtils: topk
using JSON: JSON
using ..Metalhead: Metalhead, ArchSpec, ConfigFormat, register!, declared_args
using ..Metalhead: AbstractProcessor, AbstractImageProcessor, ProcessorFiles, SourceFile,
                   register_processor!, processor_from_entries, read_json, has_file,
                   jsonvalue, image_batch, convert_to_rgb, is_8bit, RESAMPLE_FILTERS,
                   check_inputs, @processor
using ..Timm: dropout_or_identity, attention_fdrop, feature_take_indices, stage_outputs,
              pil_resize
import ..Metalhead: layer_params, architecture_of, model_args
import ..Metalhead: forward_features, forward_head, forward_intermediates
import ..Metalhead: load_processor, processor_components, ignored_keys, restricted_keys,
                    legacy_keys, check_processor, processor_fields

include("config.jl")
include("layers.jl")
include("llama.jl")
include("qwen2.jl")
include("qwen3.jl")
include("gemma3.jl")
include("granite.jl")
include("granitemoe.jl")
include("qwen3_5.jl")
include("qwen3_5_vision.jl")
include("tokenization_auto.jl")
include("tokenization_utils_tokenizers.jl")
include("image_processing_backends.jl")
include("image_processing_qwen2_vl.jl")
include("clip.jl")
include("sam3.jl")
include("image_processing_sam3.jl")
include("processing_sam3.jl")

end # module
