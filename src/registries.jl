using FeatureRegistries: Registry, Field

commafiy(n::Integer) =
         join(reverse(join.(reverse.(Iterators.partition(digits(n), 3)))), ',')

function model_registry()
  registry = Registry((;
        serial_number = Field(Integer, name = "Serial Number", description = "Serial number"),
        model_name = Field(
            String,
            name = "Model name",
            description = "The name of the model",
        ),
        parameters = Field(
            Any,
            name = "Parameters",
            description = "The number of parameters in the model",
        )),
        name = "Models", id = :model_name,)

  for (i, name) in enumerate(_MODEL_NAMES)
    param_string = Base.Docs.doc(Base.Docs.Binding(Main, name))
    push!(registry, (serial_number = i, model_name = String(name), parameters = param_string))
  end

  return registry
end
