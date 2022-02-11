struct BlockParams
  repeat::Int
  kernel::Tuple{Int, Int}
  stride::Int
  expansion_ratio::Int
  in_channels::Int
  out_channels::Int
end

struct GlobalParams
  width_coef::Real
  depth_coef::Real
  image_size::Tuple{Int, Int}

  depth_divisor::Int
  min_depth::Union{Nothing, Int}
end

# (width_coefficient, depth_coefficient, resolution)
get_efficientnet_coefficients(model_name::String) =
  Dict(
    "b0" => (1.0, 1.0, 224),
    "b1" => (1.0, 1.1, 240),
    "b2" => (1.1, 1.2, 260),
    "b3" => (1.2, 1.4, 300),
    "b4" => (1.4, 1.8, 380),
    "b5" => (1.6, 2.2, 456),
    "b6" => (1.8, 2.6, 528),
    "b7" => (2.0, 3.1, 600),
    "b8" => (2.2, 3.6, 672))[model_name]

function get_efficientnet_params(model_name)
  block_params = [
    BlockParams(1, (3, 3), 1, 1,  32,  16),
    BlockParams(2, (3, 3), 2, 6,  16,  24),
    BlockParams(2, (5, 5), 2, 6,  24,  40),
    BlockParams(3, (3, 3), 2, 6,  40,  80),
    BlockParams(3, (5, 5), 1, 6,  80, 112),
    BlockParams(4, (5, 5), 2, 6, 112, 192),
    BlockParams(1, (3, 3), 1, 6, 192, 320)]

  width_coef, depth_coef, resolution = get_efficientnet_coefficients(model_name)
  global_params = GlobalParams(
    width_coef, depth_coef, (resolution, resolution), 8, nothing)
  block_params, global_params
end

function round_filter(filters, global_params::GlobalParams)
  global_params.width_coef ≈ 1 && return filters

  depth_divisor = global_params.depth_divisor
  filters *= global_params.width_coef
  min_depth = global_params.min_depth
  min_depth = min_depth ≡ nothing ? depth_divisor : min_depth

  new_filters = max(min_depth, (floor(Int, filters + depth_divisor / 2) ÷ depth_divisor) * depth_divisor)
  new_filters < 0.9 * filters && (new_filters += global_params.depth_divisor)
  new_filters
end
