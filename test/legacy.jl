using Metalhead, Test
using Flux

deprecated(name) = (:warn, Regex("^`\\Q$(nameof(name))\\E` is deprecated"))

@testset "deprecation warnings" begin
    # invalid arguments, so that the constructors fail right after warning
    for (f, arg) in [(VGG, 0), (ResNet, 0), (WideResNet, 0), (ResNeXt, 0), (DenseNet, 0),
        (SEResNet, 0), (SEResNeXt, 0), (Res2Net, 0), (Res2NeXt, 0), (MNASNet, :none),
        (ConvMixer, :none), (ConvNeXt, :none), (MLPMixer, :none), (ResMLP, :none),
        (gMLP, :none), (ViT, :none), (EfficientNet, :none), (EfficientNetv2, :none),
        (MobileNetv3, :none)]
        @test_logs deprecated(f) @test_throws AssertionError f(arg)
    end
    for f in [AlexNet, GoogLeNet, Inceptionv3, Inceptionv4, InceptionResNetv2, Xception,
        SqueezeNet, MobileNetv1, MobileNetv2]
        @test_logs deprecated(f) @test_throws ArgumentError f(; inchannels = -1)
    end

    # the deprecated VGG(imsize; config) names the replacement, not the legacy `vgg`
    vgg16 = r"`VGG` is deprecated.*`create_model\(\"vgg16\"\)`"
    model = @test_logs (:warn, vgg16) VGG((32, 32); config = [(8, 1)], nclasses = 10)
    @test size(model(rand(Float32, 32, 32, 3, 1))) == (10, 1)

    features = r"`backbone` is deprecated.*`forward_features\(model, x\)`"
    @test (@test_logs (:warn, features) backbone(model)) === model.layers[1]
    head = r"`classifier` is deprecated.*`forward_head\(model, features\)`"
    @test (@test_logs (:warn, head) classifier(model)) === model.layers[2]
    @test_logs deprecated(Metalhead.loadpretrain!) @test_throws ArgumentError begin
        Metalhead.loadpretrain!(model, "absent")
    end

    # one warning for a call to a deprecated function that calls others
    @test_logs deprecated(MobileNetv1) @test_throws ArgumentError MobileNetv1(0.25;
        pretrain = true)
    unet = @test_logs deprecated(UNet) UNet((32, 32))
    @test unet isa UNet

    # once per session
    logger = Test.TestLogger()
    Base.CoreLogging.with_logger(logger) do
        backbone(model)
        return backbone(model)
    end
    @test length(logger.logs) == 1

    # reconstructing a model doesn't call the deprecated constructors
    @test_logs fmap(identity, model)
    @test_logs Flux.f64(model)
end

@testset "replacements" begin
    replacements = filter(!isnothing, collect(values(Metalhead.LEGACY_REPLACEMENTS)))
    architectures = [m[1] for r in replacements for m in eachmatch(r"\"(\w+)\"", r)]
    @test length(architectures) > 30
    @test setdiff(architectures, list_models()) == []
    new = [:Metalhead, :create_model, :create_processor, :list_models, :forward_features,
        :forward_head, :forward_intermediates, :feature_info,
        :post_process_object_detection, :post_process_instance_segmentation,
        :post_process_semantic_segmentation]
    @test setdiff(names(Metalhead), new, keys(Metalhead.LEGACY_REPLACEMENTS)) == []
end
