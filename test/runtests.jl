using Metalhead, Test

# Standardized testing for the models of tomorrow
@info "Starting Basic Models Tests..."
@testset "Basic Model Tests" begin
    for (T, MODEL) in [
            (Float32, VGG19),
            (Float32, SqueezeNet),
            (Float64, DenseNet),
            (Float64, GoogleNet),
        ]
	@info "Testing $(MODEL)..."
        model = MODEL()

        x_test = rand(T, 224, 224, 3, 1)
        y_test = model(x_test)

        # Test that types and shapes work out as we expect
        @test y_test isa AbstractArray
        @test length(y_test) == 1000

        # Test that the models can be indexed
        @test length(model.layers[1:4].layers) == 4
    end
end

# Test proper download and functioning of CIFAR10
@info "Starting CIFAR10 tests..."
@testset "CIFAR dataset tests" begin
    x1 = trainimgs(CIFAR10)[1]
    x2 = valimgs(CIFAR10)[1]

    # Test that the input is roughly what we expect
    @test size(x1.img) == (32, 32)
    @test size(x2.img) == (32, 32)
end

# Test printing of prediction
@info "Testing Prediction on CIFAR10..."
@testset "Prediction table display" begin
    x = valimgs(CIFAR10)[1]
    m = VGG19()
    predict(m, x)
end

# Just run the prediction code end-to-end
# TODO: Set up travis to actually run these
if length(datasets()) == 2
    for dataset in (ImageNet, CIFAR10)
        val1 = valimgs(dataset)[1]
        predict(vgg19, val1)
        classify(vgg19, val1)
    end
    predict(vgg19, testimgs(dataset(ImageNet))[1])
end
