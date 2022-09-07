# Working with pre-trained models from Metalhead

Using a model from Metalhead is as simple as selecting a model from the table of [available models](@ref API-Reference). For example, below we use the pre-trained ResNet-18 model.

```@example 1
using Metalhead
  
model = ResNet(18; pretrain = true);
```

## Training

Now, we can use this model with Flux like any other model. First, let's check the accuracy on a test image from ImageNet.

```@example 1
using Images

# test image
img = Images.load(download("https://cdn.pixabay.com/photo/2015/05/07/11/02/guitar-756326_960_720.jpg"));
```

We'll use the popular [DataAugmentation.jl](https://github.com/lorenzoh/DataAugmentation.jl) library to crop our input image, convert it to a plain array, and normalize the pixels.

```@example 1
using DataAugmentation
using Flux: onecold

DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD = (0.229, 0.224, 0.225)

augmentations = CenterCrop((224, 224)) |>
                ImageToTensor() |>
                Normalize(DATA_MEAN, DATA_STD)

data = apply(augmentations, Image(img)) |> itemdata

# ImageNet labels
labels = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))

onecold(model(Flux.unsqueeze(data, 4)), labels)
```

Below, we train it on some randomly generated data.

```@example 1
using Optimisers
using Flux: onehotbatch
using Flux.Losses: logitcrossentropy

batchsize = 1
data = [(rand(Float32, 224, 224, 3, batchsize), Flux.onehotbatch(rand(1:1000, batchsize), 1:1000))
        for _ in 1:3]
opt = Optimisers.Adam()
state = Optimisers.setup(rule, model);  # initialise this optimiser's state
for (i, (image, y)) in enumerate(data)
    @info "Starting batch $i ..."
    gs, _ = gradient(model, image) do m, x  # calculate the gradients
        logitcrossentropy(m(x), y)
    end;
    state, model = Optimisers.update(state, model, gs);
end
```

## Using pre-trained models as feature extractors

Metalhead provides the [`backbone`](@ref) and [`classifier`](@ref) functions to extract the backbone and classifier of a pre-trained model, respectively. This is useful if you want to use the pre-trained model as a feature extractor.

```@example 1