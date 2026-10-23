# Compare a port with the reference outputs of a real checkpoint, written by dump_reference.py.
#
#   julia --project=.. check_reference.jl ../test/references/timm--<repo>.safetensors ...
#
# The checkpoint is loaded from the Python Hub cache that dump_reference.py filled, so this needs
# no network access from Julia.
using Metalhead, SafeTensors
include(joinpath(@__DIR__, "..", "test", "parity.jl"))
using .Parity

function hub_cache()
    return get(ENV, "HF_HUB_CACHE",
        joinpath(get(ENV, "HF_HOME", joinpath(homedir(), ".cache", "huggingface")), "hub"))
end

pass = true
for file in ARGS
    meta = SafeTensors.deserialize(file).metadata
    ref = Metalhead.Checkpoint(file)
    snapshot = joinpath(hub_cache(), "models--" * replace(meta["repo"], "/" => "--"),
        "snapshots", meta["revision"])
    model = create_model(snapshot; pretrained = true)
    x = reference_input(Tuple(Metalhead.JSON.parse(meta["input_shape"])))
    logits = relerr(model(x), ref["logits"])
    pre_logits = relerr(forward_head(model, forward_features(model, x); pre_logits = true), ref["pre_logits"])
    ok = logits <= 1e-4 && pre_logits <= 1e-4
    global pass &= ok
    println(
        rpad(meta["repo"], 50), " logits ", round(logits; sigdigits = 2), "  pre_logits ",
        round(pre_logits; sigdigits = 2), ok ? "  ok" : "  FAIL")
end
exit(pass ? 0 : 1)
