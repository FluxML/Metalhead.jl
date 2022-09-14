# [Working with pre-trained models from Metalhead](@id pretrained)

Using a model from Metalhead is as simple as selecting a model from the table of [available models](@ref API-Reference). For example, below we use the pre-trained ResNet-18 model.

```@example 1
using Metalhead
  
model = ResNet(18; pretrain = true);
```

## Using pre-trained models as feature extractors

The `backbone` and `classifier` functions do exactly what their names suggest - they are used to extract the backbone and classifier of a model respectively. For example, to extract the backbone of a pre-trained ResNet-18 model:

```@example 1
backbone(model);
```

The `backbone` function could also be useful for people looking to just use specific sections of the model for transfer learning. The function returns a `Chain` of the layers of the model, so you can easily index into it to get the layers you want. For example, to get the first five layers of a pre-trained ResNet model,
you can just write `backbone(model)[1:5]`.

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
using Flux
using Flux: onecold

DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD = (0.229, 0.224, 0.225)

augmentations = CenterCrop((224, 224)) |>
                ImageToTensor() |>
                Normalize(DATA_MEAN, DATA_STD)

data = apply(augmentations, Image(img)) |> itemdata

# ImageNet labels
labels = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))

onecold(model(Flux.unsqueeze(data, 4)), labels);
```

That is fairly accurate! Below, we train the model on some randomly generated data:

```@example 1
using Optimisers
using Flux: onehotbatch
using Flux.Losses: logitcrossentropy

batchsize = 1
data = [(rand(Float32, 224, 224, 3, batchsize), onehotbatch(rand(1:1000, batchsize), 1:1000))
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
