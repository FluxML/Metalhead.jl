# Compare a port with the reference outputs of a real checkpoint, written by dump_reference.py.
#
#   julia --project=.. check_reference.jl ../test/references/<org>--<repo>.safetensors ...
#
# Metalhead shares the Hub cache of Python's huggingface_hub, so the checkpoint that
# dump_reference.py downloaded loads with no network access.
using Metalhead, SafeTensors
include(joinpath(@__DIR__, "..", "test", "parity.jl"))
using .Parity

pass = true
for file in ARGS
    meta = SafeTensors.deserialize(file).metadata
    ref = Metalhead.Checkpoint(file)
    if startswith(get(meta, "class", ""), "CLIP")
        r = check_clip_reference(file)
        errs = [k => v for (k, v) in pairs(r) if v isa Float64]
        ok = all(<=(1e-4) ∘ last, errs) && get(r, :processor_same, true)
        global pass &= ok
        println(rpad(meta["repo"], 50),
            join(["$k $(round(v; sigdigits = 2))" for (k, v) in errs], "  "),
            haskey(r, :processor_same) ?
            "  processor " * (r.processor_same ? "same" : "DIFFERENT") : "",
            ok ? "  ok" : "  FAIL")
        continue
    end
    if haskey(meta, "class")  # a diffusers model or pipeline
        r = check_diffusers_reference(file)
        errs = [k => v for (k, v) in pairs(r) if k != :repo]
        ok = all(<=(1e-3) ∘ last, errs)
        global pass &= ok
        println(rpad(meta["repo"], 50), join(["$k $(round(v; sigdigits = 2))" for (k, v) in errs], "  "),
            ok ? "  ok" : "  FAIL")
        continue
    end
    if haskey(ref, "text/pred_logits")  # SAM 3
        r = check_sam3_reference(file)
        errs = [k => v for (k, v) in pairs(r) if v isa Float64]
        ok = all(<=(1e-4) ∘ last, errs) && r.post_same
        global pass &= ok
        println(rpad(meta["repo"], 50), join(["$k $(round(v; sigdigits = 2))" for (k, v) in errs], "  "),
            "  post-processing ", r.post_same ? "same" : "DIFFERENT", ok ? "  ok" : "  FAIL")
        continue
    end
    if haskey(ref, "input_ids")
        model = reference_model(file)
        checks = Pair{String, Function}["" => check_lm_reference]
        haskey(ref, "image/input_ids") && push!(checks, " (image)" => check_image_reference)
        for (what, check) in checks
            r = check(file; model)
            ok = r.hidden_states <= 1e-4 && r.last_logits <= 1e-4 && r.generated
            global pass &= ok
            println(rpad(meta["repo"] * what, 50), " hidden_states ",
                round(r.hidden_states; sigdigits = 2),
                "  last_logits ", round(r.last_logits; sigdigits = 2),
                "  greedy ", r.generated ? "same" : "DIFFERENT", ok ? "  ok" : "  FAIL")
        end
        continue
    end
    model = create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
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
