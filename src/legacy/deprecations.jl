# The replacements of the API deprecated in 0.10 (see docs/src/migration.md), or `nothing`
const LEGACY_REPLACEMENTS = Dict{Symbol, Union{Nothing, String}}(
    :AlexNet => nothing,
    :VGG => "`create_model(\"vgg16\")` (or `\"vgg16_bn\"` with batch normalisation, ...)",
    :ResNet => "`create_model(\"resnet50\")` (or `\"resnet18\"`, `\"resnet101\"`, ...)",
    :WideResNet => "`create_model(\"wide_resnet50_2\")` (or `\"wide_resnet101_2\"`)",
    :ResNeXt => "`create_model(\"resnext50_32x4d\")` (or `\"resnext101_32x8d\"`, ...)",
    :DenseNet => "`create_model(\"densenet121\")` (or `\"densenet161\"`, ...)",
    :GoogLeNet => nothing,
    :Inceptionv3 => "`create_model(\"inception_v3\")`",
    :Inceptionv4 => "`create_model(\"inception_v4\")`",
    :InceptionResNetv2 => "`create_model(\"inception_resnet_v2\")`",
    :Xception => "the closest architecture, `create_model(\"xception41\")`,",
    :SEResNet => "`create_model(\"seresnet50\")` (or `\"seresnet101\"`, ...)",
    :SEResNeXt => "`create_model(\"seresnext50_32x4d\")` " *
                  "(or `\"seresnext101_32x4d\"`, ...)",
    :Res2Net => "`create_model(\"res2net50_26w_4s\")` (or `\"res2net101_26w_4s\"`, ...)",
    :Res2NeXt => "`create_model(\"res2next50\")`",
    :SqueezeNet => nothing,
    :MobileNetv1 => "`create_model(\"mobilenetv1_100\")` (or `\"mobilenetv1_125\"`)",
    :MobileNetv2 => "`create_model(\"mobilenetv2_100\")` (or `\"mobilenetv2_140\"`, ...)",
    :MobileNetv3 => "`create_model(\"mobilenetv3_large_100\")` " *
                    "(or `\"mobilenetv3_small_100\"`)",
    :MNASNet => "`create_model(\"mnasnet_100\")` for `:B1` " *
                "(or `\"semnasnet_100\"` for `:A1`)",
    :EfficientNet => "`create_model(\"efficientnet_b0\")` " *
                     "(or `\"tf_efficientnet_b0\"`, ...)",
    :EfficientNetv2 => "`create_model(\"tf_efficientnetv2_s\")` " *
                       "(or `\"tf_efficientnetv2_m\"`, ...)",
    :ConvMixer => "`create_model(\"convmixer_768_32\")` (or `\"convmixer_1536_20\"`)",
    :ConvNeXt => "`create_model(\"convnext_tiny\")` (or `\"convnext_small\"`, ...)",
    :MLPMixer => "`create_model(\"mixer_b16_224\")` (or `\"mixer_s16_224\"`, ...)",
    :ResMLP => "`create_model(\"resmlp_12_224\")` (or `\"resmlp_24_224\"`, ...)",
    :gMLP => "`create_model(\"gmlp_s16_224\")` (or `\"gmlp_ti16_224\"`, ...)",
    :ViT => "`create_model(\"vit_base_patch16_224\")` (or `\"vit_tiny_patch16_224\"`, ...)",
    :UNet => nothing,
    :backbone => "`forward_features(model, x)`",
    :classifier => "`forward_head(model, features)`",
    :loadpretrain! => "`create_model(\"<organisation>/<repository>\")`")

# Warns once per session for each deprecated function (Base.depwarn is silent outside of
# tests). Deprecated functions call each other only through internal functions that don't
# warn, so that a call warns once.
function legacy_depwarn(name::Symbol)
    replacement = LEGACY_REPLACEMENTS[name]
    use = replacement === nothing ? ", with no replacement" : ": use $replacement instead"
    msg = "`$name` is deprecated and will be removed in Metalhead 0.11$use. See the " *
          "migration guide, https://fluxml.ai/Metalhead.jl/dev/migration/."
    @warn msg maxlog = 1 _id = Symbol(:Metalhead_, name)
    return nothing
end

# Deprecated; to be removed in a future release
function VGG(imsize::Dims{2}; config, batchnorm::Bool = false, dropout_prob = 0.5,
    inchannels::Integer = 3, nclasses::Integer = 1000)
    legacy_depwarn(:VGG)
    layers = vgg(imsize; config, inchannels, batchnorm, nclasses, dropout_prob)
    return VGG(layers)
end
