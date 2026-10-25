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

The processor of a language model is the tokenizer of its repository, which
[`create_processor`](@ref) loads as a
[`TokenizersBackend`](@ref Metalhead.Transformers.TokenizersBackend), with the special
tokens, padding and maximum length of its `tokenizer_config.json`. It turns texts into the
inputs of the model, and [`decode`](@ref Metalhead.Transformers.decode) turns token indices
back into text. Strings need
[HuggingFaceTokenizers.jl](https://github.com/MurrellGroup/HuggingFaceTokenizers.jl), which
tokenizes them with the repository's `tokenizer.json`:

```julia
using HuggingFaceTokenizers
using Metalhead.Transformers: decode

tok = create_processor("HuggingFaceTB/SmolLM2-135M")

prompt = "The capital of France is Paris, and the capital of Italy is"
ids = tok(prompt).input_ids
for _ in 1:8                                      # greedy decoding
    push!(ids, argmax(model(ids)[:, end]))
end
println(decode(tok, ids))
```

```
The capital of France is Paris, and the capital of Italy is Rome.

The capital of the
```

Token indices start at 1, as for Flux's `Embedding`: the tokenizer returns the ids of
Hugging Face's tokenizers plus 1, and `decode` takes such indices. Without
HuggingFaceTokenizers, the tokenizer still takes token ids, from 0 as a tokenizer gives
them, and pads them.

There is no key-value cache yet, so each step of this loop runs the model on the whole
sequence.

## Batches

A vector of texts makes a batch: token indices of size `(seq_len, batch_size)`, padded to
the longest text with the tokenizer's padding token, and an `attention_mask` that is 0 for
the padding. The model takes the output of the tokenizer as it is:

```julia
model = create_model("Qwen/Qwen3-0.6B"; pretrained = true)
tok = create_processor("Qwen/Qwen3-0.6B")

inputs = tok(["The capital of France is", "Hello"])
inputs.attention_mask          # [1 1; 1 0; 1 0; 1 0; 1 0]
logits = model(inputs)         # 151936 × 5 × 2
decode(tok, inputs.input_ids; skip_special_tokens = true)
# ["The capital of France is", "Hello"]
```

The tokenizer pads on its side, `tok.padding_side`: on the right for SmolLM2 and Qwen, on
the left for Llama 3, Gemma 3 and Granite. The model takes either, and gives each text the
logits it has alone, at the positions of its tokens. Padded on the left, the last position
holds the last token of every text, from which to generate:

```julia
inputs = tok(["The capital of France is", "Hello"]; padding_side = "left")
next = model(inputs)[:, end, :]                     # the logits of the next tokens
[decode(tok, [argmax(l)]) for l in eachcol(next)]   # [" Paris", " Answer"]
```

A tokenizer without a padding token, such as SmolLM2's, can't pad, as in transformers. Set
one when creating it:
`create_processor("HuggingFaceTB/SmolLM2-135M"; pad_token = "<|endoftext|>")`. Likewise,
`add_bos_token` and `add_eos_token` set which of the BOS and EOS tokens surround each text,
and `show(tok)` prints each setting and where it comes from.

## Inputs and outputs

- `model(ids)` takes token indices of size `(seq_len, batch_size)`, or a vector for a single
  sequence, and returns logits of size `(vocab_size, seq_len, batch_size)` (or
  `(vocab_size, seq_len)`). `model(inputs)` takes the output of the tokenizer.
- Attention is causal. A batch padded on the right takes the tokenizer's `attention_mask`,
  of size `(seq_len, batch_size)`, as a keyword argument: `model(ids; attention_mask)`. The
  padding takes no part in the attention of the other tokens. `model(inputs)` also takes a
  batch padded on the left.
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
expert, so the experts compute only for their own tokens, as in transformers. Grouping the
tokens by expert copies the experts of each token to the CPU, once per layer on a GPU. The
load-balancing loss of training is not implemented.
