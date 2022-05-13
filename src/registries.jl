using FeatureRegistries: Registry, Field

commafiy(n::Integer) =
         join(reverse(join.(reverse.(Iterators.partition(digits(n), 3)))), ',')

function _modelregistry()
  registry = Registry((;
        serial_number = Field(Integer, name = "Serial Number", description = "Serial number"),
        model_name = Field(
            String,
            name = "Model name",
            description = "The name of the model",
        ),
        parameters = Field(
            String,
            name = "Parameters",
            description = "The number of parameters in the model",
        )),
        name = "Models", id = :model_name,)

  for (i, name) in enumerate(_MODEL_NAMES)
    model = @eval $name()
    ps = Flux.params(model)
    params = sum(length, ps)
    nonparams = Flux._childarray_sum(length, model) - params
    param_string = commafiy(params) * " trainable parameters\n" * commafiy(nonparams) *
                   " non-trainable parameters."
    push!(registry, (serial_number = i, model_name = String(name), parameters = param_string))
  end

  return registry
end

const MODELS = _modelregistry()
