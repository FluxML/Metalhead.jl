using Metalhead, Flux, Test, InteractiveUtils

run(`ulimit -a`)

# Standardized testing for the models of tomorrow
@testset "Untrained Model Tests" begin
    for (T, MODEL) in [
            (Float64, VGG11),
            (Float64, VGG13),
            (Float64, VGG16),
            (Float64, VGG19),
            (Float64, ResNet18),
            (Float64, ResNet34),
            (Float64, ResNet50),
            (Float64, ResNet101),
            (Float64, ResNet152),
            (Float64, DenseNet121),
            (Float64, DenseNet169),
            (Float64, DenseNet201),
            (Float64, DenseNet264),
            (Float64, GoogleNet)
        ]
        GC.gc()

        model = MODEL()
        model = Flux.mapleaves(Flux.Tracker.data, model)

        x_test = rand(T, 224, 224, 3, 1)
        y_test = model(x_test)

        # Test that types and shapes work out as we expect
        @test y_test isa AbstractArray
        @test length(y_test) == 1000

        # Test that the models can be indexed
        @test length(model.layers[1:4].layers) == 4

        # Make all the allocations nothing for GC to free them
        model = nothing
        x_test = nothing
        y_test = nothing
    end
    GC.gc()
    # Test if batchnorm models work properly
    for (T, MODEL) in [
            (Float64, VGG19),
            (Float64, VGG16),
            (Float64, VGG13),
            (Float64, VGG11)
        ]
        GC.gc()

        model = MODEL(true)
        model = Flux.mapleaves(Flux.Tracker.data, model)

        x_test = rand(T, 224, 224, 3, 1)
        y_test = model(x_test)

        # Test that types and shapes work out as we expect
        @test y_test isa AbstractArray
        @test length(y_test) == 1000

        # Test that the models can be indexed
        @test length(model.layers[1:4].layers) == 4

        # Make all the allocations nothing for GC to free them
        model = nothing
        x_test = nothing
        y_test = nothing
    end
    GC.gc()
    # Test models which have a version parameter
    for (T, version, MODEL) in [
            (Float64, "1.0",SqueezeNet),
            (Float64, "1.1",SqueezeNet)
        ]
        GC.gc()

        model = MODEL(version)
        model = Flux.mapleaves(Flux.Tracker.data, model)

        x_test = rand(T, 224, 224, 3, 1)
        y_test = model(x_test)

        # Test that types and shapes work out as we expect
        @test y_test isa AbstractArray
        @test length(y_test) == 1000

        # Test that the models can be indexed
        @test length(model.layers[1:4].layers) == 4

        # Make all the allocations nothing for GC to free them
        model = nothing
        x_test = nothing
        y_test = nothing
    end
    GC.gc()
end

@testset "Trained Model Tests" begin
    for (T, MODEL) in [
            (Float32, VGG19),
            (Float32, SqueezeNet),
            (Float64, ResNet50),
            (Float64, DenseNet121),
            (Float64, GoogleNet)
        ]
        GC.gc()

        model = trained(MODEL)
        model = Flux.mapleaves(Flux.Tracker.data, model)

        x_test = rand(T, 224, 224, 3, 1)
        y_test = model(x_test)

        # Test that types and shapes work out as we expect
        @test y_test isa AbstractArray
        @test length(y_test) == 1000

        # Test that the models can be indexed
        @test length(model.layers[1:4].layers) == 4

        # Make all the allocations nothing for GC to free them
        model = nothing
        x_test = nothing
        y_test = nothing
    end
end

# Test proper download and functioning of CIFAR10
@testset "CIFAR dataset tests" begin
    x1 = trainimgs(CIFAR10)[1]
    x2 = valimgs(CIFAR10)[1]

    # Test that the input is roughly what we expect
    @test size(x1.img) == (32, 32)
    @test size(x2.img) == (32, 32)
end

# Test printing of prediction
@testset "Prediction table display" begin
    x = valimgs(CIFAR10)[1]
    m = trained(VGG19)
    predict(m, x)
end

# Just run the prediction code end-to-end
# TODO: Set up travis to actually run these
if length(datasets()) == 2
    vgg19 = trained(VGG19)
    for dataset in (ImageNet, CIFAR10)
        val1 = valimgs(dataset)[1]
        predict(vgg19, val1)
        classify(vgg19, val1)
    end
    predict(vgg19, testimgs(dataset(ImageNet))[1])
end
