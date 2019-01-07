# Common argument parsing and validation
using DocOpt

"""
    mod_lookup(mod, name)

Attempt to lookup `name` within the module `mod`.  If that fails, attempt to
look it up in `Main`, and if that fails throw an error.
"""
function mod_lookup(mod, name)
    try
        # First, try to lookup `name` from within `mod`:
        getfield(mod, Symbol(name))
    catch
        # If that doesn't work, try from within `Main`:
        try
            getfield(Main, Symbol(name))
        catch
            error("Could not dynamically resolve $(name) within $(mod) or Main!")
        end
    end
end

"""
    parse_option_list(optlist::String)

Given a string, split it on commas, then attempt to parse each value as first
a Float64, then as a value within `Main`, then finally keeping it as a string.
If it has an `=` character within it, map everything before that character to
a `String` and parse the second half normally, returning a `Pair` for use as a
keyword argument later on.
"""
function parse_option_list(optlist::String)
    # Split `"decay=1e-3"` into `"decay" => "1e-3"`
    function split_kwargs(data)
        if occursin('=', data)
            k, v = split(data, "=")[1:2]
            return k => v
        end
        return data
    end

    # Give automatic data type parsing the old college try.
    parsify(data::Pair) = (data.first => floatify(data.second))
    function parsify(data)
        # First, try parsing it as a float
        try
            return parse(Float64, data)
        catch
        end

        # If that doesn't work, try interpreting it as a globally-accessible value
        try
            return getfield(Main, Symbol(data))
        catch
        end

        # If nothing else works, just keep it as a string
        return data
    end

    # Split optlist into an actual, comma-separated list, then split
    # kwargs out into Pair's before parsifying the whole thing:
    opts = parsify.(split_kwargs.(split(optlist, ",")))

    # Now extract the positional and keyword arguments, returning both:
    args = filter(o -> !(typeof(o) <: Pair), opts)
    kwargs = filter(o -> typeof(o) <: Pair, opts)
    return args, kwargs
end


# Have DocOpt do the actual parsing
args = docopt(doc, version=v"2.0.0")

# Utility functions
arg_set(name) = !in(get(args, name, nothing), (false, nothing))


## Directory Options
if arg_set("--train-dir")
    train_data_dir = args["--train-dir"]
    @assert isdir(train_data_dir) "Training data '$(train_data_dir)' must exist!"
end

if arg_set("--val-dir")
    val_data_dir = args["--val-dir"]
    @assert isdir(val_data_dir) "Validation data '$(val_data_dir)' must exist!"
end

if arg_set("--output-dir")
    output_data_dir = args["--output-dir"]
    if !isdir(output_data_dir)
        mkpath(output_data_dir)
    end
end

continue_training = arg_set("--continue")


## Training options
if arg_set("--batch-size")
    batch_size = parse(Int, args["--batch-size"])
    @assert batch_size >= 1 "Batch size must be a non-negative integer"
end

if arg_set("--max-epochs")
    max_epochs = parse(Int, args["--max-epochs"])
    @assert max_epochs >= 1 "Max epochs must be a non-negative integer"
end

if arg_set("--patience")
    patience = parse(Int, args["--patience"])
    # If patience is -1, then just set it to the maximum number of epochs + 1
    if patience == -1
        patience = max_epochs + 1
    end
    @assert patience >= 1 "Patience must be a non-negative integer or -1"
end

if arg_set("--model-opts")
    model_args, model_kwargs = parse_option_list(args["--model-opts"])
else
    # Initialize opt_args/opt_kwargs so that we can construct our optimizer
    # later even if no options have been set
    model_args = ()
    model_kwargs = ()
end

# Get our model constructor
if arg_set("--model")
    model_name = args["--model"]
    model_ctor(args...; kwargs...) = mod_lookup(Metalhead, model_name)(args...; kwargs...)
end


if arg_set("--opt-opts")
    opt_args, opt_kwargs = parse_option_list(args["--opt-opts"])
else
    # Initialize opt_args/opt_kwargs so that we can construct our optimizer
    # later even if no options have been set
    opt_args = ()
    opt_kwargs = ()
end

if arg_set("--opt")
    opt_name = args["--opt"]
    opt_ctor(args...; kwargs...) = mod_lookup(Flux.Optimise, opt_name)(args...; kwargs...)
end

## Performance options
if arg_set("--data-workers")
    data_workers = parse(Int, args["--data-workers"])
    if data_workers == -1
        data_workers = Sys.CPU_THREADS
    end
    @assert data_workers >= 1 "Data workers must be a non-negative integer or -1"
end

if arg_set("--gpus")
    # roflmao
    ENV["CUDA_VISIBLE_DEVICES"] = args["--gpus"]
    gpu_devices = parse.(Int, split(args["--gpus"], ","))
else
    gpu_devices = Int[]
end
