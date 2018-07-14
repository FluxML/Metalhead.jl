abstract type ClassificationModel{Class} end
labels(model::Type{<:ClassificationModel{Class}}) where {Class} = labels(Class)
labels(model::ClassificationModel{Class}) where {Class} = labels(Class)

function trained(which)
  if which == VGG11
    error("Pretrained Weights for VGG11 are not available")
  elseif which == VGG11_BN
    error("Pretrained Weights for VGG11_BN are not available")
  elseif which == VGG13
    error("Pretrained Weights for VGG13 are not available")
  elseif which == VGG13_BN
    error("Pretrained Weights for VGG13_BN are not available")
  elseif which == VGG16
    error("Pretrained Weights for VGG16 are not available")
  elseif which == VGG16_BN
    error("Pretrained Weights for VGG16_BN are not available")
  elseif which == VGG19
    VGG19(trained_vgg19_layers())
  elseif which == VGG19_BN
    error("Pretrained Weights for VGG19_BN are not available")
  end
end
