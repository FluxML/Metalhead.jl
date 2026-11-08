# The config.json of a transformers repository: its `model_type`, the class in `architectures`,
# and the fields of the model's config class, at the top level, among generic keys (token ids,
# generation settings, `transformers_version`, ...) that the models don't read.
struct TransformersFormat <: AbstractConfigFormat end
const TRANSFORMERS = TransformersFormat()
direct_weight_init(::TransformersFormat) = true
Base.show(io::IO, ::TransformersFormat) = print(io, "transformers")

# The class that transformers' AutoModel builds for a `model_type` (`MODEL_MAPPING_NAMES`),
# of the ported classes: the architecture of a config.json without `architectures`, as
# those of the SigLIP 2 checkpoints of fixed resolution.
const MODEL_TYPE_CLASSES = Dict{String, String}()

function architecture_of(::TransformersFormat, config)
    haskey(config, "model_type") || return nothing
    archs = get(config, "architectures", nothing)
    archs isa AbstractVector && !isempty(archs) ||
        return get(MODEL_TYPE_CLASSES, config["model_type"], nothing)
    length(archs) == 1 ||
        error("config.json names several architectures, $(join(archs, ", ")); expected one")
    return only(archs)
end

# The keys of config.json that every transformers model ignores: what `PretrainedConfig`
# records about the checkpoint, the settings of training, fine-tuning and generation that
# older versions saved in it, and the ids of tokens that no ported model reads (a model that
# does, such as CLIP's `eos_token_id`, declares the key). A model with an output head
# declares `tie_word_embeddings`.
const IGNORED_CONFIG_KEYS = Set([
    # what is recorded about the checkpoint
    "architectures", "model_type", "transformers_version", "_name_or_path", "name_or_path",
    "_commit_hash", "auto_map", "dtype", "torch_dtype", "tokenizer_class", "unsloth_fixed",
    # the runtime: Metalhead picks the element type, and the attention
    "use_cache", "output_hidden_states", "output_attentions", "return_dict",
    "chunk_size_feed_forward", "gradient_checkpointing", "torchscript", "use_bfloat16",
    "tf_legacy_loss", "attn_implementation", "_attn_implementation",
    "experts_implementation", "pruned_heads",
    # fine-tuning: the heads that need the labels declare them
    "id2label", "label2id", "problem_type", "finetuning_task", "task_specific_params",
    "prefix",
    # the output head, which the models that have one declare
    "tie_word_embeddings",
    # the ids of special tokens
    "bos_token_id", "eos_token_id", "pad_token_id", "sep_token_id", "cls_token_id",
    "decoder_start_token_id", "forced_bos_token_id", "forced_eos_token_id",
    # encoder-decoders
    "is_encoder_decoder", "is_decoder", "add_cross_attention",
    "cross_attention_hidden_size", "tie_encoder_decoder",
    # generation, which `generation_config.json` sets: older versions saved it here
    "max_length", "min_length", "do_sample", "early_stopping", "num_beams",
    "num_beam_groups", "diversity_penalty", "temperature", "top_k", "top_p", "typical_p",
    "repetition_penalty", "length_penalty", "no_repeat_ngram_size",
    "encoder_no_repeat_ngram_size", "bad_words_ids", "num_return_sequences",
    "output_scores", "return_dict_in_generate", "remove_invalid_values",
    "exponential_decay_length_penalty", "suppress_tokens", "begin_suppress_tokens"])

ignores_config_key(::TransformersFormat, key) = key in IGNORED_CONFIG_KEYS

# The keys of config.json that the builder declares as keywords, as transformers ignores the
# others, overridden by `kwargs`. Keys that neither the builder nor the format or the
# architecture ignore warn.
function model_args(format::TransformersFormat, spec, config, kwargs)
    return merge(spec.defaults, merge_configs(declared_args(format, spec, config), kwargs))
end

# `overrides` over `config`, where an override of a sub-config (`text_config`, ...) is merged
# into the sub-config of `config` instead of replacing it, so that its other fields keep the
# values of the checkpoint rather than fall back to those of the config class.
function merge_configs(config::NamedTuple, overrides::NamedTuple)
    merged = merge(config, overrides)
    for key in keys(overrides)
        endswith(string(key), "_config") || continue
        old, new = get(config, key, nothing), overrides[key]
        old isa NamedTuple && new isa NamedTuple || continue
        merged = merge(merged, NamedTuple{(key,)}((merge_configs(old, new),)))
    end
    return merged
end
