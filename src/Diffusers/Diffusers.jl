"""
    Metalhead.Diffusers

Flux ports of [Hugging Face Diffusers](https://github.com/huggingface/diffusers) models and
pipelines. Each ported model mirrors the PyTorch module tree, so that pretrained `state_dict`s
load by name. [`Metalhead.create_model`](@ref) builds a whole pipeline from a repository, such
as [`Flux2KleinPipeline`](@ref Metalhead.Diffusers.Flux2KleinPipeline) from
`"black-forest-labs/FLUX.2-klein-4B"`, or one of its components with `subfolder`.

Images are WHCN arrays with values in [-1, 1] for the autoencoders, and [0, 1] for the output
of pipelines.
"""
module Diffusers

using Flux
using NNlib
using Random: Random, AbstractRNG
using ConcreteStructs: @concrete
using ChainRulesCore: ChainRulesCore
using JSON: JSON
using ..Metalhead: ArchSpec, AbstractConfigFormat, ModelSource, register!, declared_args,
                   init_layer,
                   identify, lookup, source_config, source_file, subsource, ProcessorFiles,
                   load_tokenizer_or_defer, resolve_tokenizer, DeferredTokenizer,
                   encode_ids, pad_token_id, pad_ids
using ..Transformers: RMSNorm, to_device_of
import ..Metalhead: architecture_of, model_args, weights_name, instantiate,
                    instantiate_processor, forward_intermediates, direct_weight_init

include("config.jl")
include("layers.jl")
include("vae.jl")
include("autoencoder_kl_flux2.jl")
include("transformer_flux2.jl")
include("scheduler.jl")
include("pipeline_flux2_klein.jl")

end # module
