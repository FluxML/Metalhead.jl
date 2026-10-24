# Ported from transformers.models.granitemoe.modeling_granitemoe,
# transformers.models.granitemoeshared.modeling_granitemoeshared and
# transformers.models.granitemoehybrid.modeling_granitemoehybrid (transformers 5.20,
# Apache-2.0, The Hugging Face Team), without the Mamba layers of the hybrid models.
# Validated against test/fixtures/granitemoe*_ and test/references/ibm-granite--* and
# ibm-research--moe-7b-1b-active-shared-experts.
#
# The structs mirror the keys of the checkpoints, which transformers 5 renames on loading
# (`conversion_mapping.py`): the experts of `block_sparse_moe` are `input_linear.weight` and
# `output_linear.weight` (its `experts.gate_up_proj` and `experts.down_proj`), and the
# router's Linear layer is `router.layer` (its `router.weight`).

# The experts' stacked weights, torch's (num_experts, out, in) as (in, out, num_experts).
@concrete struct GraniteMoeParallelExperts
    weight
end
Flux.@layer GraniteMoeParallelExperts

# `GraniteMoeTopKRouter`: the `top_k` experts of each token by logit, weighted by the
# softmax of their logits, which transformers computes in float32.
@concrete struct GraniteMoeTopKGating
    layer
    top_k
end
Flux.@layer GraniteMoeTopKGating

function (m::GraniteMoeTopKGating)(x)
    T = eltype(x)
    logits, index = topk(m.layer(x), m.top_k)
    return index, T.(softmax(promote_type(T, Float32).(logits); dims = 1))
end

@concrete struct GraniteMoeMoE
    router
    input_linear
    output_linear
    activation
end
Flux.@layer GraniteMoeMoE

function (m::GraniteMoeMoE)(x)
    tokens = reshape(x, size(x, 1), :)
    index, weights = m.router(tokens)
    y = moe_experts(m.input_linear.weight, m.output_linear.weight, m.activation, tokens,
        index, weights)
    return reshape(y, size(x))
end

# The shared expert of GraniteMoeShared and GraniteMoeHybrid, a gated MLP.
@concrete struct GraniteMoeSharedMLP
    input_linear
    output_linear
    activation
end
Flux.@layer GraniteMoeSharedMLP

(m::GraniteMoeSharedMLP)(x) = m.output_linear(gated(m.activation, m.input_linear(x)))

# `GraniteDecoderLayer` whose MLP is the sum of a sparse mixture of experts
# `block_sparse_moe` and a shared expert `shared_mlp`, either of which may be `nothing`.
@concrete struct GraniteMoeDecoderLayer
    self_attn
    block_sparse_moe
    shared_mlp
    input_layernorm
    post_attention_layernorm
    residual_multiplier
end
Flux.@layer GraniteMoeDecoderLayer

function (m::GraniteMoeDecoderLayer)(x, position_embeddings, kv_lengths = nothing)
    r = eltype(x)(m.residual_multiplier)
    x = x .+ m.self_attn(m.input_layernorm(x), position_embeddings, kv_lengths) .* r
    return x .+ feedforward(m, m.post_attention_layernorm(x)) .* r
end

function feedforward(m::GraniteMoeDecoderLayer, x)
    m.shared_mlp === nothing && return m.block_sparse_moe(x)
    m.block_sparse_moe === nothing && return m.shared_mlp(x)
    return m.block_sparse_moe(x) .+ m.shared_mlp(x)
end

"""
    granitemoe_for_causal_lm(; vocab_size = 32000, hidden_size = 4096, num_hidden_layers = 32, kwargs...)

Port of transformers' `GraniteMoeForCausalLM` (IBM Granite 3 MoE), built from the fields of
its `GraniteMoeConfig` with the same names and defaults, and the keys of
[`granite_for_causal_lm`](@ref Metalhead.Transformers.granite_for_causal_lm) that it adds
(`rope_theta`, `rope_scaling`, `head_dim`). It takes and returns the arrays of
[`llama_for_causal_lm`](@ref Metalhead.Transformers.llama_for_causal_lm).

Granite MoE is Granite whose MLP is a sparse mixture of `num_local_experts` gated MLPs of
`intermediate_size` features: each token goes through the `num_experts_per_tok` experts of
largest router logit, whose outputs are weighted by the softmax of these logits. The training
options (`output_router_logits`, `router_aux_loss_coef`) are not supported.
"""
function granitemoe_for_causal_lm(; vocab_size = 32000, hidden_size = 4096,
        intermediate_size = 11008, num_hidden_layers = 32, num_attention_heads = 32,
        num_key_value_heads = nothing, head_dim = nothing, hidden_act = "silu",
        max_position_embeddings = 2048, initializer_range = 0.02, rms_norm_eps = 1e-6,
        tie_word_embeddings = false, rope_parameters = nothing, rope_theta = nothing,
        rope_scaling = nothing, attention_bias = false, attention_dropout = 0.0,
        embedding_multiplier = 1.0, logits_scaling = 1.0, residual_multiplier = 1.0,
        attention_multiplier = 1.0, num_local_experts = 8, num_experts_per_tok = 2)
    return granitemoe_like("GraniteMoeForCausalLM"; vocab_size, hidden_size,
        intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
        head_dim, hidden_act, max_position_embeddings, initializer_range, rms_norm_eps,
        tie_word_embeddings, rope_parameters, rope_theta, rope_scaling, attention_bias,
        attention_dropout, embedding_multiplier, logits_scaling, residual_multiplier,
        attention_multiplier, num_local_experts, num_experts_per_tok,
        shared_intermediate_size = nothing)
end

"""
    granitemoeshared_for_causal_lm(; vocab_size = 32000, hidden_size = 4096, num_hidden_layers = 32, kwargs...)

Port of transformers' `GraniteMoeSharedForCausalLM`: Granite MoE (see
[`granitemoe_for_causal_lm`](@ref Metalhead.Transformers.granitemoe_for_causal_lm)) with a
shared expert, a gated MLP of `shared_intermediate_size` features that every token goes
through, unless `shared_intermediate_size = 0`.
"""
function granitemoeshared_for_causal_lm(; vocab_size = 32000, hidden_size = 4096,
        intermediate_size = 11008, num_hidden_layers = 32, num_attention_heads = 32,
        num_key_value_heads = nothing, head_dim = nothing, hidden_act = "silu",
        max_position_embeddings = 2048, initializer_range = 0.02, rms_norm_eps = 1e-6,
        tie_word_embeddings = false, rope_parameters = nothing, rope_theta = nothing,
        rope_scaling = nothing, attention_bias = false, attention_dropout = 0.0,
        embedding_multiplier = 1.0, logits_scaling = 1.0, residual_multiplier = 1.0,
        attention_multiplier = 1.0, num_local_experts = 8, num_experts_per_tok = 2,
        shared_intermediate_size = 0)
    return granitemoe_like("GraniteMoeSharedForCausalLM"; vocab_size, hidden_size,
        intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
        head_dim, hidden_act, max_position_embeddings, initializer_range, rms_norm_eps,
        tie_word_embeddings, rope_parameters, rope_theta, rope_scaling, attention_bias,
        attention_dropout, embedding_multiplier, logits_scaling, residual_multiplier,
        attention_multiplier, num_local_experts, num_experts_per_tok,
        shared_intermediate_size = shared_intermediate_size == 0 ? nothing :
                                   shared_intermediate_size)
end

"""
    granitemoehybrid_for_causal_lm(; vocab_size = 32000, hidden_size = 4096, num_hidden_layers = 32, kwargs...)

Port of transformers' `GraniteMoeHybridForCausalLM` (IBM Granite 4.0) without its Mamba
layers: the models whose `layer_types` are all `"attention"` (or `"full_attention"`), such as
`granite-4.0-350m` and `granite-4.0-1b`. A layer is that of
[`granitemoeshared_for_causal_lm`](@ref Metalhead.Transformers.granitemoeshared_for_causal_lm),
whose shared expert is always present, with no mixture of experts if
`num_local_experts = 0`. The attention has a rotary embedding only with
`position_embedding_type = "rope"`. Mamba layers (`"mamba"` or `"linear_attention"`, the
default) error, and the `mamba_*` keys of `config.json` are ignored.
"""
function granitemoehybrid_for_causal_lm(; vocab_size = 32000, hidden_size = 4096,
        intermediate_size = 11008, num_hidden_layers = 32, num_attention_heads = 32,
        num_key_value_heads = nothing, head_dim = nothing, hidden_act = "silu",
        max_position_embeddings = 2048, initializer_range = 0.02, rms_norm_eps = 1e-6,
        tie_word_embeddings = false, rope_parameters = nothing, rope_theta = nothing,
        rope_scaling = nothing, attention_bias = false, attention_dropout = 0.0,
        embedding_multiplier = 1.0, logits_scaling = 1.0, residual_multiplier = 1.0,
        attention_multiplier = 1.0, num_local_experts = 8, num_experts_per_tok = 2,
        shared_intermediate_size = 1024, position_embedding_type = nothing,
        layer_types = nothing)
    name = "GraniteMoeHybridForCausalLM"
    types = something(layer_types, fill("linear_attention", num_hidden_layers))
    length(types) == num_hidden_layers ||
        error("`layer_types` has $(length(types)) entries, `num_hidden_layers` is $num_hidden_layers")
    for t in types
        t in ("mamba", "linear_attention") &&
            error("Mamba layers (`layer_types` entry $(repr(t))) are not supported by the Metalhead port of $name: only attention layers are")
        t in ("attention", "full_attention") ||
            error("`layer_types` entry $(repr(t)) is not supported by the Metalhead port of $name")
    end
    return granitemoe_like(name; vocab_size, hidden_size, intermediate_size,
        num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, hidden_act,
        max_position_embeddings, initializer_range, rms_norm_eps, tie_word_embeddings,
        rope_parameters, rope_theta, rope_scaling, attention_bias, attention_dropout,
        embedding_multiplier, logits_scaling, residual_multiplier, attention_multiplier,
        num_local_experts, num_experts_per_tok, shared_intermediate_size,
        rope = position_embedding_type == "rope", dense_layers = true)
end

# A Granite model whose layers have `num_local_experts` experts, none if 0 (only with
# `dense_layers`), and a shared expert of `shared_intermediate_size` features, none if
# `nothing`.
function granitemoe_like(name; vocab_size, hidden_size, intermediate_size,
        num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, hidden_act,
        max_position_embeddings, initializer_range, rms_norm_eps, tie_word_embeddings,
        rope_parameters, rope_theta, rope_scaling, attention_bias, attention_dropout,
        embedding_multiplier, logits_scaling, residual_multiplier, attention_multiplier,
        num_local_experts, num_experts_per_tok, shared_intermediate_size, rope = true,
        dense_layers = false)
    num_local_experts >= (dense_layers ? 0 : 1) ||
        error("`num_local_experts` ($num_local_experts) must be positive")
    num_local_experts == 0 || 1 <= num_experts_per_tok <= num_local_experts ||
        error("`num_experts_per_tok` ($num_experts_per_tok) must be in 1:$num_local_experts, `num_local_experts`")
    params = rope_parameters_of(name; rope_theta, rope_scaling, rope_parameters,
        max_position_embeddings)
    act = act_fn(hidden_act)
    init = normal_init(initializer_range)
    function decoder_layer(attention, dense, norm)
        moe = num_local_experts == 0 ? nothing :
              GraniteMoeMoE(
            GraniteMoeTopKGating(dense(hidden_size => num_local_experts, false),
                num_experts_per_tok),
            GraniteMoeParallelExperts(init(hidden_size, 2 * intermediate_size,
                num_local_experts)),
            GraniteMoeParallelExperts(init(intermediate_size, hidden_size,
                num_local_experts)),
            act)
        shared = shared_intermediate_size === nothing ? nothing :
                 GraniteMoeSharedMLP(
            dense(hidden_size => 2 * shared_intermediate_size, false),
            dense(shared_intermediate_size => hidden_size, false), act)
        return GraniteMoeDecoderLayer(attention, moe, shared, norm(), norm(),
            Float64(residual_multiplier))
    end
    return granite_like(decoder_layer; vocab_size, hidden_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, head_dim, initializer_range,
        rms_norm_eps, tie_word_embeddings, rope_parameters = rope ? params : nothing,
        attention_bias, attention_dropout, embedding_multiplier, logits_scaling,
        attention_multiplier)
end

register!(ArchSpec(TRANSFORMERS, "GraniteMoeForCausalLM", granitemoe_for_causal_lm;
    family = "granite",
    references = [
        "ibm-granite/granite-3.1-1b-a400m-base" => "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"]))

register!(ArchSpec(TRANSFORMERS, "GraniteMoeSharedForCausalLM",
    granitemoeshared_for_causal_lm;
    family = "granite",
    references = [
        "ibm-research/moe-7b-1b-active-shared-experts" => "6194d7218c3e792b939b535f5640f9c32be9b442"]))

register!(ArchSpec(TRANSFORMERS, "GraniteMoeHybridForCausalLM",
    granitemoehybrid_for_causal_lm;
    family = "granite",
    references = [
        "ibm-granite/granite-4.0-350m-base" => "a50b46cef21c8a86b15f0496cb794487a78a910b"]))
