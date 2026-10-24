"""
    Metalhead.Transformers

Flux ports of [Hugging Face Transformers](https://github.com/huggingface/transformers) models.
Each ported model mirrors the PyTorch module tree, so that pretrained `state_dict`s load by
name; build them with [`Metalhead.create_model`](@ref) from a Hub repository.

Language models take token indices starting at 1, as Flux's `Embedding` does: add 1 to the
ids of a Hugging Face tokenizer, and subtract 1 from the index of a logit to get its token id.
"""
module Transformers

using Flux
using NNlib
using ConcreteStructs: @concrete
using ChainRulesCore: ChainRulesCore
using ..Metalhead: ArchSpec, ConfigFormat, register!, declared_args
using ..Timm: dropout_or_identity, attention_fdrop, feature_take_indices, stage_outputs
import ..Metalhead: layer_params, architecture_of, model_args
import ..Metalhead: forward_features, forward_head, forward_intermediates

include("config.jl")
include("layers.jl")
include("llama.jl")
include("qwen2.jl")
include("qwen3.jl")
include("gemma3.jl")
include("granite.jl")
include("granitemoe.jl")

end # module
