"""
    Metalhead.SentenceTransformers

Flux ports of [sentence-transformers](https://github.com/huggingface/sentence-transformers)
models, the text embeddings of the Hub: the modules of a repository's `modules.json`
(`Transformer`, `Pooling`, `Dense`, `Normalize`) over the ported transformers models,
with the settings of its configuration files, read strictly.
[`Metalhead.create_model`](@ref) builds a
[`SentenceTransformer`](@ref Metalhead.SentenceTransformers.SentenceTransformer) from a
repository with a modules.json, such as `"Qwen/Qwen3-Embedding-0.6B"`.
"""
module SentenceTransformers

using Flux: Flux, relu, sigmoid, gelu_erf
using NNlib: NNlib
using ConcreteStructs: @concrete
using ChainRulesCore: ChainRulesCore
using JSON: JSON
using ..Metalhead: ArchSpec, AbstractConfigFormat, ModelSource, register!, lookup,
                   model_args, subsource, source_name, source_file_if_exists,
                   source_weights, Checkpoint, load_state!, DEFAULT_IGNORE,
                   ELTYPE_CONVERTERS, build_pretrained, ProcessorFiles, read_json,
                   load_tokenizer_or_defer, resolve_tokenizer, DeferredTokenizer,
                   encode_ids, pad_token_id, pad_ids, text_list, check_inputs, jsonvalue,
                   warn_unknown
using ..Transformers: TRANSFORMERS, CausalLM, DecoderModel, decoder
import ..Metalhead: architecture_of, instantiate, instantiate_processor

include("config.jl")
include("transformer.jl")
include("pooling.jl")
include("dense.jl")
include("normalize.jl")
include("model.jl")

end # module
