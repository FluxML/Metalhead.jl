# [Language models](@id language_models)

Metalhead also ports language models from
[Hugging Face Transformers](https://github.com/huggingface/transformers), starting with the
Llama, Qwen, Gemma 3 and Granite architectures (`LlamaForCausalLM`, `Qwen2ForCausalLM`,
`Qwen3ForCausalLM`, `Gemma3ForCausalLM`, `GraniteForCausalLM`, `GraniteMoeForCausalLM`,
`GraniteMoeSharedForCausalLM` and `GraniteMoeHybridForCausalLM`).
They load from the Hub like the timm models:

```julia
using Metalhead

model = create_model("HuggingFaceTB/SmolLM2-135M"; pretrained = true)
```

Any Hub repository whose `config.json` names a ported architecture loads, with the same strict
checks. See the [list of models](@ref models) for the architectures and the checkpoints they
were validated against.

## Generating text

Metalhead has no tokenizers. [HuggingFaceTokenizers.jl](https://github.com/MurrellGroup/HuggingFaceTokenizers.jl)
loads the tokenizer of a Hub repository:

```julia
using HuggingFaceTokenizers

tokenizer = from_pretrained(Tokenizer, "HuggingFaceTB/SmolLM2-135M")

prompt = "The capital of France is Paris, and the capital of Italy is"
ids = encode(tokenizer, prompt).ids .+ 1          # Metalhead token indices start at 1
for _ in 1:8                                      # greedy decoding
    push!(ids, argmax(model(ids)[:, end]))
end
println(decode(tokenizer, ids .- 1))
```

```
The capital of France is Paris, and the capital of Italy is Rome.

The capital of the
```

Token indices start at 1, as for Flux's `Embedding`: add 1 to the ids of a Hugging Face
tokenizer, and subtract 1 from the index of a logit to get its token id.

There is no key-value cache yet, so each step of this loop runs the model on the whole
sequence.

## Inputs and outputs

- `model(ids)` takes token indices of size `(seq_len, batch_size)`, or a vector for a single
  sequence, and returns logits of size `(vocab_size, seq_len, batch_size)` (or
  `(vocab_size, seq_len)`).
- Attention is causal. Sequences of different lengths are batched by padding them on the right
  and passing the tokenizer's `attention_mask`, of size `(seq_len, batch_size)`, as a keyword
  argument: `model(ids; attention_mask)`. The padding takes no part in the attention of the
  other tokens.
- [`forward_features`](@ref) returns the hidden states after the final norm, of size
  `(hidden_size, seq_len, batch_size)`, and [`forward_head`](@ref) applies the output
  projection `lm_head` to them.
- [`forward_intermediates`](@ref) returns the hidden states of the decoder layers, which
  diffusion models such as FLUX.2 use as text embeddings:
  `forward_intermediates(model, ids; indices = [9, 18, 27], intermediates_only = true)`. The
  `i`-th is transformers' `output_hidden_states[i]`.
- The weights are converted to `eltype` (`Float32` by default) when they are loaded, whatever
  their type on the Hub (often `bfloat16`).

## Configuration

The keyword arguments of `create_model` are the fields of the model's `config.json`, with the
names and defaults of transformers. With a repository, they override its configuration:

```julia
create_model("HuggingFaceTB/SmolLM2-135M"; num_hidden_layers = 4)   # random weights
create_model("LlamaForCausalLM"; vocab_size = 1000, hidden_size = 64, intermediate_size = 128,
    num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2)
```

The rotary embeddings are the default ones, or those scaled as in Llama 3.1 and later
(`rope_type = "llama3"`) or linearly (`"linear"`). Options that the ports don't implement raise
an error. These are the other rotary embeddings, such as `"yarn"`, and the sliding-window
attention of Qwen2 and Qwen3 (`use_sliding_window`), which their released checkpoints leave
off. Gemma 3 does have sliding-window attention: its layers attend either to all the previous
tokens or to the last `sliding_window` of them, as its `layer_types` say. Granite 4.0
(`GraniteMoeHybridForCausalLM`) loads only without its Mamba layers, as `granite-4.0-350m`
and `granite-4.0-1b` are; the `-h-` models have them.

transformers derives the Qwen2 and Qwen3 classes from Llama's, and their ports reuse the
Llama structs: a Qwen model is a `LlamaForCausalLM`, whose Qwen3 attention layers are
`Qwen3Attention`s. A Gemma 3 model is a `Gemma3ForCausalLM` whose decoder is a `LlamaModel` of
`Gemma3DecoderLayer`s, and the four Granite classes are `GraniteForCausalLM`s.

## Mixtures of experts

In Granite MoE (`GraniteMoeForCausalLM`, and `GraniteMoeSharedForCausalLM` with a shared
expert), the MLP of each layer is a sparse mixture of experts: a router picks the
`num_experts_per_tok` experts of each token, whose outputs are summed with the router's
weights. Each projection of the experts is one `batched_mul` over the tokens grouped by
expert, so the experts compute only for their own tokens, as in transformers. The router
picks its experts on the CPU, so a model on the GPU copies the router's logits to the CPU
once per layer. The load-balancing loss of training is not implemented.
