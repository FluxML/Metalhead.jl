"""
    backbone(model)

This function returns the backbone of a model that can be used for feature extraction.
A `Flux.Chain` is returned, which can be indexed/sliced into to get the desired layer(s).
Note that the model used here as input must be the "camel-cased" version of the model,
e.g. `ResNet` instead of `resnet`.
"""
backbone

"""
    classifier(model)

This function returns the classifier head of a model. This is sometimes useful for fine-tuning
a model on a different dataset. A `Flux.Chain` is returned, which can be indexed/sliced into to
get the desired layer(s). Note that the model used here as input must be the "camel-cased"
version of the model, e.g. `ResNet` instead of `resnet`.
"""
classifier
