# [Vision-language models](@id vision_language)

Qwen3.5, of Alibaba's Qwen team, reads images and text, and answers in text. Metalhead ports
its models, `Qwen3_5ForConditionalGeneration`, with their vision encoder, and their
processor, [`Qwen3VLProcessor`](@ref Metalhead.Transformers.Qwen3VLProcessor), which turns
images and prompts into the inputs of the model.

## Set up

```julia
using Pkg
Pkg.add(["Metalhead", "FileIO", "ImageIO", "ImageCore", "HuggingFaceTokenizers"])
```

```julia
using Metalhead, FileIO, ImageCore, HuggingFaceTokenizers
using Downloads: download

model = create_model("Qwen/Qwen3.5-0.8B"; pretrained = true)
proc = create_processor("Qwen/Qwen3.5-0.8B")
```

The processor takes text with the tokenizer of the repository, which `create_processor`
loads after `using HuggingFaceTokenizers`.

## Prompts

The processor doesn't apply the chat template of the model: write the prompt as the template
writes it, with `<|vision_start|><|image_pad|><|vision_end|>` where the image goes. The empty
block of thoughts, `<think>\n\n</think>`, makes the model answer without reasoning first.

```julia
chat(question) = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" *
                 question * "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

img = load(download("http://images.cocodataset.org/val2017/000000039769.jpg"))   # 640 × 480
inputs = proc(; images = img, text = chat("What is between the two cats?"))
inputs.image_grid_thw          # 3 × 1: (1, 30, 40), 30 rows of 40 patches
length(inputs.input_ids)       # 321 tokens, of which 300 stand for the image
```

![Two cats asleep on a pink couch, with two remote controls](../assets/cats.jpg)

*The [photograph of COCO](http://images.cocodataset.org/val2017/000000039769.jpg) that the
prompts ask about.*

The processor resizes each image to sides that are multiples of 32 pixels (this one already
is), cuts it into the patches of 16 × 16 pixels that the model takes, and replaces the
placeholder `<|image_pad|>` with one image token for each 2 × 2 patches. Each placeholder of
the prompts stands for the next image, in order.

## Answering

The model takes the inputs as they are, and returns the logits of the next token at each
position. Greedy decoding appends the most probable token until the end of the answer,
`<|im_end|>`:

```julia
(; input_ids, pixel_values, image_grid_thw) = inputs
im_end = only(encode(proc.tokenizer, "<|im_end|>").ids) + 1   # token indices start at 1
n = length(input_ids)
for _ in 1:200
    next = argmax(model(input_ids; pixel_values, image_grid_thw)[:, end])
    next == im_end && break
    push!(input_ids, next)
end
println(decode(proc.tokenizer, input_ids[(n + 1):end] .- 1))
```

```
Between the two cats, there is a **white remote control** lying on the pink couch.

- The left cat is sleeping with its head near the remote.
- The right cat is also sleeping, and its head is near the same remote.
- The remote appears to be a standard TV remote (with buttons for channels, power, etc.), though it’s not clearly identifiable as a specific brand.

So, the answer is: **A white remote control**.
```

There is no key-value cache yet, so each step runs the model on the whole sequence: the 97
steps of this answer take 6 seconds on an RTX 5090, and a few minutes on a CPU. Asked to
`"Describe this image."`, the model starts with "This image captures a serene and cozy scene
of two cats lounging on a bright pink couch, surrounded by their favorite remote controls."

A vector of prompts makes a batch, padded on the right with an `attention_mask`, which the
model takes as it is. See [`Qwen3VLProcessor`](@ref Metalhead.Transformers.Qwen3VLProcessor)
for the inputs it returns, and
[`Qwen3_5ForConditionalGeneration`](@ref Metalhead.Transformers.Qwen3_5ForConditionalGeneration)
for the model.
