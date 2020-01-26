// Copyright 2020 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Definition and implementation of the neural network's layers.

use std::num::NonZeroUsize;
use std::ops::Add;

use crate::matrix::Matrix;
use crate::Error;
use crate::Result;

/// A layer of the neural network.
///
/// The layer can be used as an input, hidden, or output layer.
#[derive(Debug)]
pub struct Layer {
    /// The weights for this layer's input.
    ///
    /// This is a `o x i` matrix where `o` is the number of this layer's output nodes and `i`
    /// the number of input nodes.
    weights: Matrix<f64>,

    /// The bias of this layer.
    ///
    /// This is a `o x 1` matrix where `o` is the number of this layer's output nodes.
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
    /// [`Error::DimensionsTooLarge`]: ../enum.Error.html#variant.DimensionsTooLarge
    pub fn new(input_nodes: NonZeroUsize, output_nodes: NonZeroUsize) -> Result<Layer> {
        // Weights are `o x i`.
        let weights = Matrix::from_random(output_nodes, input_nodes)?;

        // Bias is `o x 1`.
        let bias = Matrix::from_random(output_nodes, NonZeroUsize::new(1).unwrap())?;

        Ok(Layer { weights, bias })
    }

    // endregion

    // region AI

    /// Predict an output of this layer for the given input.
    ///
    /// The input matrix must be an `i x 1` matrix where `i` is the number of (input) nodes in this
    /// layer. Otherwise, [`Error::DimensionMismatch`] will be returned.
    ///
    /// The output matrix will be a `o x 1` matrix where `o` is the number of outputs of this layer.
    ///
    /// [`Error::DimensionMismatch`]: ../enum.Error.html#variant.DimensionMismatch
    pub fn predict(&self, input: Matrix<f64>) -> Result<Matrix<f64>> {
        // The input matrix must have only one column.
        if input.get_columns() != 1 {
            return Err(Error::DimensionMismatch);
        }

        // Multiply the input to the weights (using matrix multiplication), then add the bias.
        let mut output: Matrix<f64> = self.weights.matrix_mul(&input)?;

        // Explicitly call `add` instead of using the operator so it is more legible with the try
        // operator `?`.
        output = output.add(&self.bias)?;

        // Apply the activation function.
        output.map(|element, _row, _column| 1.0 / (1.0 + (-element).exp()));

        Ok(output)
    }

    // endregion
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::num::NonZeroUsize;

    use crate::Error;

    // region Initialization

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

        // The bias is `output x 1`, i.e. `3x1`.
        assert_eq!(layer.bias.get_rows(), output_nodes.get());
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

    // endregion

    // region AI

    /// Test the prediction of this layer with valid dimensions.
    #[test]
    fn predict_valid_dimensions() {
        let one = NonZeroUsize::new(1).unwrap();
        let input_nodes = NonZeroUsize::new(3).unwrap();
        let output_nodes = NonZeroUsize::new(2).unwrap();

        // Create a layer, but for testing, use known weights and biases.
        let mut layer = Layer::new(input_nodes, output_nodes).unwrap();
        layer.weights.map(|_element, _row, _column| 0.5);
        layer.bias.map(|_element, _row, _column| 0.1);

        let input: Matrix<f64> = Matrix::from_slice(input_nodes, one, &[1.0, 1.1, 1.2]).unwrap();
        let prediction_result: Result<Matrix<f64>> = layer.predict(input);
        assert!(prediction_result.is_ok());

        let prediction: Matrix<f64> = prediction_result.unwrap();
        assert_eq!(prediction.get_rows(), output_nodes.get());
        assert_eq!(prediction.get_columns(), 1);
        assert_eq!(
            prediction.as_slice(),
            &[0.851_952_801_968_310_6, 0.851_952_801_968_310_6]
        );
    }

    /// Test the prediction of this layer if the input matrix has too many columns.
    #[test]
    fn predict_too_many_input_columns() {
        let input_nodes = NonZeroUsize::new(3).unwrap();
        let output_nodes = NonZeroUsize::new(2).unwrap();

        let layer = Layer::new(input_nodes, output_nodes).unwrap();
        let input: Matrix<f64> = Matrix::new(input_nodes, output_nodes, 1.0).unwrap();
        let prediction_result: Result<Matrix<f64>> = layer.predict(input);
        assert!(prediction_result.is_err());

        let is_correct_error: bool = match prediction_result.unwrap_err() {
            Error::DimensionMismatch => true,
            _ => false,
        };

        assert!(
            is_correct_error,
            "Expected error Error::DimensionMismatch not satisfied."
        );
    }

    /// Test the prediction of this layer if the input matrix has the wrong number of rows.
    #[test]
    fn predict_wrong_number_of_input_rows() {
        let one = NonZeroUsize::new(1).unwrap();
        let input_nodes = NonZeroUsize::new(3).unwrap();
        let output_nodes = NonZeroUsize::new(2).unwrap();

        let layer = Layer::new(input_nodes, output_nodes).unwrap();
        let input: Matrix<f64> = Matrix::new(output_nodes, one, 1.0).unwrap();
        let prediction_result: Result<Matrix<f64>> = layer.predict(input);
        assert!(prediction_result.is_err());

        let is_correct_error: bool = match prediction_result.unwrap_err() {
            Error::DimensionMismatch => true,
            _ => false,
        };

        assert!(
            is_correct_error,
            "Expected error Error::DimensionMismatch not satisfied."
        );
    }

    // endregion
}
