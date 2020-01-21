// Copyright 2020 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Definition and implementation of the neural network's layers.

use std::num::NonZeroUsize;

use crate::matrix::Matrix;
use crate::Result;

/// A layer of the neural network.
///
/// The layer can be used as an input, hidden, or output layer.
#[derive(Debug)]
struct Layer {
    /// The weights for this layer's input.
    ///
    /// This is a `o x i` matrix where `o` is the number of this layer's output nodes and `i`
    /// the number of input nodes.
    weights: Matrix<f64>,

    /// The bias of this layer.
    ///
    /// This is a `i x 1` matrix where `i` is the number of this layer's input nodes.
    bias: Matrix<f64>,
}

impl Layer {
    // region Initialize

    /// Create a new layer within a neural network. The layer will have the given number of input
    /// and output nodes.
    ///
    /// The weights and bias will be initialized with random values within `[0.0, 1.0]`.
    ///
    /// The product of the number of input nodes and output nodes must not exceed the maximum
    /// `usize` value, [`::std::usize::MAX`]. Otherwise, an [`Error::DimensionsTooLarge`] will be
    /// returned.
    ///
    /// [`::std::usize::MAX`]: https://doc.rust-lang.org/stable/std/usize/constant.MAX.html
    /// [`Error::DimensionsTooLarge`]: enum.Error.html#variant.DimensionsTooLarge
    pub fn new(input_nodes: NonZeroUsize, output_nodes: NonZeroUsize) -> Result<Layer> {
        // Weights are `o x i`.
        let weights = Matrix::from_random(output_nodes, input_nodes)?;

        // Bias is `i x 1`.
        let bias = Matrix::from_random(input_nodes, NonZeroUsize::new(1).unwrap())?;

        Ok(Layer { weights, bias })
    }

    // endregion
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::num::NonZeroUsize;

    use crate::Error;

    /// Test creating a new layer when the size does not exceed the maximum size.
    #[test]
    fn new_valid_size() {
        let input_nodes = NonZeroUsize::new(2).unwrap();
        let output_nodes = NonZeroUsize::new(3).unwrap();

        let layer_result: Result<Layer> = Layer::new(input_nodes, output_nodes);
        assert!(layer_result.is_ok());

        let layer: Layer = layer_result.unwrap();

        // The weights are `output x input`, i.e. `3x2`.
        assert_eq!(layer.weights.get_rows(), output_nodes.get());
        assert_eq!(layer.weights.get_columns(), input_nodes.get());

        // The bias is `input x 1`, i.e. `2x1`.
        assert_eq!(layer.bias.get_rows(), input_nodes.get());
        assert_eq!(layer.bias.get_columns(), 1);
    }

    /// Test creating a new layer when the size exceeds the maximum size.
    #[test]
    fn new_invalid_size() {
        let input_nodes = NonZeroUsize::new(::std::usize::MAX).unwrap();
        let output_nodes = NonZeroUsize::new(2).unwrap();

        let layer_result: Result<Layer> = Layer::new(input_nodes, output_nodes);
        assert!(layer_result.is_err());

        let is_correct_error: bool = match layer_result.unwrap_err() {
            Error::DimensionsTooLarge => true,
            _ => false,
        };

        assert!(
            is_correct_error,
            "Expected error Error::DimensionsTooLarge not satisfied."
        );
    }
}
