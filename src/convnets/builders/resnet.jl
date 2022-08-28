function build_resnet(img_dims, stem, get_layers, block_repeats::AbstractVector{<:Integer},
                      connection, classifier_fn)
    # Build stages of the ResNet
    stage_blocks = cnn_stages(get_layers, block_repeats, connection)
    backbone = Chain(stem, stage_blocks...)
    # Add classifier to the backbone
    nfeaturemaps = Flux.outputsize(backbone, img_dims; padbatch = true)[3]
    return Chain(backbone, classifier_fn(nfeaturemaps))
end
