# Changelog

## [Unreleased]

### Fixed

- Fix `SqueezeNet(; pretrain = true)` with current Flux by loading plain weight arrays
  instead of deserializing an old Flux model ([#384](https://github.com/FluxML/Metalhead.jl/issues/384)).
