"""
    mlpblock(planes, hidden_planes; dropout = 0., dense = Dense, activation = gelu)

Feedforward block used in many vision transformer-like models.

# Arguments
- `planes`: Number of dimensions in the input and output.
- `hidden_planes`: Number of dimensions in the intermediate layer.
- `dropout`: Dropout rate.
- `dense`: Type of dense layer to use in the feedforward block.
- `activation`: Activation function to use.
"""
function mlpblock(planes, hidden_planes; dropout = 0., dense = Dense, activation = gelu)
  Chain(dense(planes, hidden_planes, activation), Dropout(dropout),
        dense(hidden_planes, planes), Dropout(dropout))
end