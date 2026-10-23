# Ported from diffusers.schedulers.scheduling_flow_match_euler_discrete (diffusers 0.42.0,
# Apache-2.0, The HuggingFace Team), without the stochastic sampling and the Karras,
# exponential and beta schedules of the sigmas.

"""
    FlowMatchEulerDiscreteScheduler

diffusers' Euler sampler of flow-matching models, as configured by the `scheduler_config.json` of
a pipeline. Unlike diffusers' scheduler it keeps no state: [`set_timesteps`](@ref
Metalhead.Diffusers.set_timesteps) returns the schedule, and [`step`](@ref
Metalhead.Diffusers.step) takes the noise levels of a step.
"""
@concrete struct FlowMatchEulerDiscreteScheduler
    num_train_timesteps
    shift
    use_dynamic_shifting
    time_shift_type
end

function flow_match_euler_discrete_scheduler(; num_train_timesteps = 1000, shift = 1.0,
        use_dynamic_shifting = false, base_shift = 0.5, max_shift = 1.15,
        base_image_seq_len = 256, max_image_seq_len = 4096, invert_sigmas = false,
        shift_terminal = nothing, use_karras_sigmas = false, use_exponential_sigmas = false,
        use_beta_sigmas = false, time_shift_type = "exponential", stochastic_sampling = false)
    for (name, value) in pairs((; invert_sigmas, use_karras_sigmas, use_exponential_sigmas,
        use_beta_sigmas, stochastic_sampling))
        value && error("`$name = true` is not supported by the Metalhead port")
    end
    shift_terminal === nothing ||
        error("`shift_terminal` is not supported by the Metalhead port")
    time_shift_type in ("exponential", "linear") ||
        error("`time_shift_type = $(repr(time_shift_type))` is not supported")
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps, Float64(shift),
        use_dynamic_shifting, time_shift_type)
end

SCHEDULERS["FlowMatchEulerDiscreteScheduler"] = flow_match_euler_discrete_scheduler

# In the precision of the sigmas `t`, as numpy computes with Python floats.
function time_shift(s::FlowMatchEulerDiscreteScheduler, mu, sigma, t)
    T = eltype(t)
    m = T(s.time_shift_type == "exponential" ? exp(mu) : mu)
    return m ./ (m .+ (1 ./ t .- 1) .^ T(sigma))
end

"""
    set_timesteps(scheduler, num_inference_steps; sigmas = nothing, mu = nothing) -> (timesteps, sigmas)

The `Float32` timesteps of the denoising steps (`sigmas .* num_train_timesteps`) and their noise
levels, with a final 0. The `sigmas` are shifted by `mu` with `use_dynamic_shifting`, which then
requires it, or else by the scheduler's `shift`. By default they are evenly spaced from 1 to
the smallest noise level of training.
"""
function set_timesteps(s::FlowMatchEulerDiscreteScheduler, num_inference_steps::Integer;
        sigmas = nothing, mu = nothing)
    s.use_dynamic_shifting && mu === nothing &&
        error("`mu` must be given with `use_dynamic_shifting = true`")
    sigmas = if sigmas === nothing
        lo = 1.0f0 / s.num_train_timesteps
        sigma_min = s.use_dynamic_shifting ? lo : shifted(s, lo)
        collect(range(1.0, Float64(sigma_min); length = num_inference_steps))
    else
        Float32.(collect(sigmas))
    end
    sigmas = Float32.(s.use_dynamic_shifting ? time_shift(s, mu, 1.0, sigmas) : shifted(s, sigmas))
    return sigmas .* s.num_train_timesteps, [sigmas; 0.0f0]
end

function shifted(s::FlowMatchEulerDiscreteScheduler, sigmas)
    T = eltype(sigmas)
    return T(s.shift) .* sigmas ./ (1 .+ T(s.shift - 1) .* sigmas)
end

"""
    step(scheduler, model_output, sigma, sigma_next, sample)

The sample of an Euler step of the flow `model_output` from the noise level `sigma` to
`sigma_next`, computed in at least `Float32` and returned in the element type of
`model_output`.
"""
function step(::FlowMatchEulerDiscreteScheduler, model_output, sigma, sigma_next, sample)
    T = promote_type(Float32, eltype(sample))
    dt = Float32(sigma_next) - Float32(sigma)
    return convert_like(model_output, T.(sample) .+ T(dt) .* model_output)
end
