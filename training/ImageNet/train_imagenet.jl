doc = """Imagenet training harness

Usage:
  train_imagenet.jl --help
  train_imagenet.jl --model=<model> --train-dir=<dir> --val-dir=<dir> [options]
  train_imagenet.jl --continue [options]


Directory options:
  --train-dir=<dir>     Location of Imagenet training data, assumed to be
                        organized as folders with ILSVRC label names each
                        containing image files.  As an example, if the file
                        `/data/train/n13044778/n13044778_621.JPEG` contains a
                        picture of an earthstar mushroom, then you should pass
                        the option `--train-dir=/data/train`.
  --val-dir=<dir>       Similar to `--train-dir` but for the validation set.
  --output-dir=<dir>    Set where models, logs and stats are saved [default: .]
                        This directory will contain a BSON file for the weights
                        of the model, a BSON file containing relevant
                        statistics over time, various logs, and a pony.
  --continue            Continue training from the output directory.

Training options:
  --model=<model>       Model to train, resolved first within `Metalhead`, then
                        resolved within `Main`.  E.g. `VGG19` will attempt to
                        first resolve as `Metalhead.VGG19`.
  --model-opts=<.,.>    Model options given in a comma-separated list, passed
                        to the constructor as positional arguments unless there
                        exists a `=` argument in the option, in which case it
                        will be passed to the model constructor as a keyword
                        argument.  To illustrate, take this example:
                        `--model-opts=256,foo,conv_type=dense,α=1e-2`
                        This will be parsed into the function call:
                        `model_ctor(256, "foo"; conv_type="dense", α=0.001)`
                        All options are first attempted to be parsed as
                        floating-point values.  If that fails, they are
                        resolved as objects accessible from `Main`.  If that
                        fails they are interpreted as strings.
  --opt=<name>          Optimiser to use for training [default: ADAM]
                        Similar to `<model>`, the optimiser is first searched
                        for within `Flux.Optimise`, but if it cannot be found
                        there, the global scope is searched.
  --opt-opts=a,b,k=v    Optimiser options in a comma-separated list,
                        interpreted in the same way as `--model-opts`
  --batch-size=<N>      Set the minibatch size [default: 64]
  --max-epochs=<N>      Set the maximum number of training epochs [default: 80]
                        An epoch is defined as running every element of the
                        training set through the model once.
  --patience=<P>        If model testing loss stops improving, [default: -1]
                        this signifies that the model has converged to a local
                        optimum.  Use `--patience` to define how many epochs of
                        no improvement must pass before giving up, or `-1` to
                        always train to the max number of epochs. [default: -1]

Performance options:
  --data-workers=<dw>   Set number of worker processes to use [default: -1]
                        for data loading.  Must be greater than or equal to 1,
                        except for the special value `-1`, which is expanded to
                        the number of cores on the local machine, as returned
                        by `Sys.CPU_THREADS`.
  --gpus=<gpus>         Engage the lesser dreadnought engine for training.
                        Identified by device ID (e.g. `--gpus=0,2,3`).
  --tpu=<xrt_addr>      Engage the greater dreadnought engine for training.
                        Identified by the XRT endpoint to communicate with,
                        (e.g. `--tpu=localhost:8740`)

Misc options:
  --help                Print out this help and exit.


Examples:
    ./train_imagenet.jl --model=VGG19 --train-dir=/data/train --val-dir=/data/val \\
                        --batch-size=64
    ./train_imagenet.jl --model=ResNet34 --train-dir=/data/train --val-dir=/data/val \\
                        --opt=ADAM --opt-opts=1e-3,decay=0.001
"""
# By including argparsing.jl, we automatically parse out the above arguments
# and define many variables, such as `train_data_dir`, `model_ctor`, `opt_args`,
# and more.
include("argparsing.jl")
include("dataset.jl")
include("training.jl")

## TODO:
#  Once https://github.com/FluxML/Flux.jl/pull/379 is merged, implement saving/loading
#  of optimizer state, so that continued training doesn't kcik out of balance.
#    - Also implement learning rate decay schedules once that work is merged.

# If we are continuing a training run, load that TrainState in rather than constructing it:
if continue_training
    ts = load_train_state(output_data_dir)
else
    # Otherwise, begin by creating our data iterators:
    train_dataset = ImagenetDataset(train_data_dir, data_workers, batch_size, imagenet_train_data_loader)
    val_dataset = ImagenetDataset(val_data_dir, data_workers, batch_size, imagenet_val_data_loader)

    # Construct model with our user-defined arguments
    model = model_ctor(model_args...; model_kwargs...)

    # Construct optimizer with our user-defined arguments
    opt = opt_ctor(params(model), opt_args...; opt_kwargs...)

    # Finally, create our TrainState object and immediately save it:
    ts = TrainState(
        model_name, model_args, model_kwargs, model,
        opt_name, opt_args, opt_kwargs, opt,
        train_dataset, val_dataset, max_epochs, patience,
    )
    @info("Saving initial model....")
    save_train_state(ts, output_data_dir)
end

# Engage the lesser dreadnaught engine
if !isempty(gpu_devices)
    Core.eval(Main, :(using CuArrays))
    @info("Mapping model onto GPU(s)...")
    ts.model = cu(ts.model)
end

# Train away, train away, train away |> 's/train/sail/ig'
@info("Beginning training run...")
train!(ts, output_data_dir)
