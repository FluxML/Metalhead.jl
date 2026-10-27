# [Zero-shot classification](@id zero_shot)

SigLIP and SigLIP 2, of Google, embed images and texts in the same space: an image and a text
that describes it have similar embeddings. They classify an image among labels that you
write, without training on them, and retrieve the images that a text describes, or the
texts that describe an image. Metalhead ports their models, `SiglipModel` and
`Siglip2Model`, and their processors, which prepare the images and the texts.

## Set up

```julia
using Pkg
Pkg.add(["Metalhead", "Flux", "FileIO", "ImageIO", "ImageCore", "HuggingFaceTokenizers"])
```

```julia
using Metalhead, Flux, FileIO, ImageCore, HuggingFaceTokenizers
using Downloads: download

model = create_model("google/siglip2-base-patch16-naflex")
proc = create_processor("google/siglip2-base-patch16-naflex")
```

The processor takes text with the tokenizer of the repository, which `create_processor`
loads after `using HuggingFaceTokenizers`; without it, the processor prepares the images
only. The NaFlex checkpoints of SigLIP 2 (`Siglip2Model`) keep the aspect ratio of an image,
which they cut into at most 256 patches of 16 × 16 pixels. The other checkpoints of SigLIP
and SigLIP 2, such as `google/siglip-base-patch16-224` and `google/siglip2-base-patch16-224`,
are `SiglipModel`s, which take square images of a fixed size; the code below is the same
for them.

For a NaFlex grid one patch wide whose height differs from the learned position grid,
PyTorch's CPU bilinear interpolation in the reference environment repeats a constant
column. Metalhead uses NNlib's bilinear interpolation, so the position embeddings differ
for that case.

## Classifying an image

Write a text for each label, and score each against the image:

```julia
img = load(download("http://images.cocodataset.org/val2017/000000039769.jpg"))   # 640 × 480
labels = ["two cats", "a dog", "a remote control", "a pink couch", "a car"]
inputs = proc(; images = img, text = ["This is a photo of $l." for l in labels])
inputs.spatial_shapes          # 2 × 1: (14, 18), 14 rows of 18 patches
size(inputs.input_ids)         # (64, 5): each text padded to 64 tokens

outputs = model(inputs)
probs = sigmoid.(outputs.logits_per_image)   # 5 × 1: one column per image
# two cats 0.416, a dog 0.000, a remote control 0.001, a pink couch 0.000, a car 0.000
```

![Two cats asleep on a pink couch, with two remote controls](../assets/cats.jpg)

*The [photograph of COCO](http://images.cocodataset.org/val2017/000000039769.jpg) that the
labels describe.*

SigLIP is trained with a sigmoid loss, which scores each pair of an image and a text alone:
`sigmoid` of a logit is the probability that the text describes the image, and the
probabilities of the labels don't sum to 1, unlike those of CLIP's softmax. A vector of
images makes a batch; `logits_per_image` has one row per text and one column per image, and
`logits_per_text` is its transpose.

The processor pads each text to 64 tokens, as SigLIP was trained. The NaFlex processor of
SigLIP 2 lowercases them first, as transformers' does; the fixed-resolution SigLIP 2
checkpoints use Gemma's tokenizer without this lowercasing. Julia's `lowercase` differs from
Python's for Unicode letters whose lowercase expands to multiple characters, such as `İ`.
The texts of `google/siglip-*` checkpoints shouldn't contain `/`, `<` or `>`: the
tokenizer.json of these repositories keeps them, while transformers' tokenizer drops them
with the other punctuation. `google/siglip-so400m-patch14-224` has only `spiece.model`, so
HuggingFaceTokenizers.jl cannot load its tokenizer; use a checkpoint with `tokenizer.json`
for text processing.

## Embeddings

`outputs.image_embeds` and `outputs.text_embeds` are the embeddings of the images and the
texts, normalized: their dot products are the cosine similarities of the logits. To search a
collection, embed its images once, and each query alone, with
[`get_image_features`](@ref Metalhead.Transformers.get_image_features) and
[`get_text_features`](@ref Metalhead.Transformers.get_text_features), which return the
outputs of the vision and text models, whose `pooler_output` is the embedding before its
normalization:

```julia
using Metalhead.Transformers: get_image_features, get_text_features
using LinearAlgebra: normalize

images = proc(; images = [img])            # the inputs of the images only
image_embeds = mapslices(normalize, get_image_features(model, images).pooler_output; dims = 1)
query = proc(; text = "Two cats asleep on a pink couch")
text_embeds = mapslices(normalize, get_text_features(model, query).pooler_output; dims = 1)
similarities = vec(text_embeds' * image_embeds)   # one per image: sort to rank them
```

See [`Siglip2Model`](@ref Metalhead.Transformers.Siglip2Model),
[`SiglipModel`](@ref Metalhead.Transformers.SiglipModel) and their processors,
[`Siglip2Processor`](@ref Metalhead.Transformers.Siglip2Processor) and
[`SiglipProcessor`](@ref Metalhead.Transformers.SiglipProcessor), for the inputs and outputs.
