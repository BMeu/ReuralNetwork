// Copyright 2020 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Definition and implementation of the neural network.

use crate::Error;
use crate::Layer;
use crate::Matrix;
use crate::Result;

// TODO: Improve documentation.
/// A neural network.
#[derive(Debug)]
pub struct NeuralNetwork {
    /// All layers of this neural network.
    ///
    /// The order of the layers within the vector is the order in which the layers will be accessed
    /// by the neural network.
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    // region Initialization

    /// Create a new neural network with the given layers.
    ///
    /// The order of the layers within the vector is the order in which the layers will be accessed
    /// by the neural network.
    ///
    /// The vector of layers must contain at least one layer. Otherwise, [`Error::EmptyNetwork`]
    /// will be returned.
    ///
    /// [`Error::EmptyNetwork`]: ../enum.Error.html#variant.EmptyNetwork
    pub(crate) fn new(layers: Vec<Layer>) -> Result<NeuralNetwork> {
        if layers.is_empty() {
            return Err(Error::EmptyNetwork);
        }

        Ok(NeuralNetwork { layers })
    }

    // endregion

    // region Getters

    /// Get a slice of all layers in the neural network.
    #[cfg(test)]
    pub(crate) fn get_layers(&self) -> &[Layer] {
        self.layers.as_slice()
    }

    // endregion

    // region AI

    /// Let the neural network predict an output for the given input.
    ///
    /// The input matrix must be an `i x 1` matrix where `i` is the number of input nodes of the
    /// neural network. Otherwise, [`Error::DimensionMismatch`] will be returned.
    ///
    /// The output matrix will be a `o x 1` matrix where `o` is the number of outputs of this layer.
    ///
    /// [`Error::DimensionMismatch`]: ../enum.Error.html#variant.DimensionMismatch
    pub fn predict(&self, input: Matrix<f64>) -> Result<Matrix<f64>> {
        // The input matrix must have only one column.
        if input.get_number_of_columns() != 1 {
            return Err(Error::DimensionMismatch);
        }

        // Let each layer predict its output, using the previous layer's output as its input.
        // The initial input is of course the input to this method. The final layer's output is the
        // output of the neural network.
        let mut output: Matrix<f64> = input;
        for layer in &self.layers {
            output = layer.predict(output)?;
        }

        Ok(output)
    }

    // endregion
}

#[cfg(test)]
mod tests {

    use super::*;

    use std::num::NonZeroUsize;

    use crate::Layer;

    // region Initialization

    /// Test creating a new neural network with enough layers.
    #[test]
    fn new_with_layers() {
        let input_nodes = NonZeroUsize::new(3).unwrap();
        let nodes_hidden_layer_1 = NonZeroUsize::new(5).unwrap();
        let nodes_hidden_layer_2 = NonZeroUsize::new(2).unwrap();
        let output_nodes = NonZeroUsize::new(1).unwrap();

        let mut layers: Vec<Layer> = Vec::with_capacity(3);
        layers.push(Layer::new(input_nodes, nodes_hidden_layer_1).unwrap());
        layers.push(Layer::new(nodes_hidden_layer_1, nodes_hidden_layer_2).unwrap());
        layers.push(Layer::new(nodes_hidden_layer_2, output_nodes).unwrap());

        let neural_network_result: Result<NeuralNetwork> = NeuralNetwork::new(layers);
        assert!(neural_network_result.is_ok());

        let neural_network: NeuralNetwork = neural_network_result.unwrap();
        assert_eq!(neural_network.layers.len(), 3);
    }

    /// Test creating a new neural network without layers.
    #[test]
    fn new_without_layers() {
        let layers: Vec<Layer> = Vec::new();
        let neural_network_result: Result<NeuralNetwork> = NeuralNetwork::new(layers);

        assert!(
            matches!(neural_network_result, Err(Error::EmptyNetwork)),
            "Expected error Error::EmptyNetwork not satisfied."
        );
    }

    // endregion

    // region Getters

    /// Test getting all layers.
    #[test]
    fn get_layers() {
        let input_nodes = NonZeroUsize::new(3).unwrap();
        let nodes_hidden_layer_1 = NonZeroUsize::new(5).unwrap();
        let nodes_hidden_layer_2 = NonZeroUsize::new(2).unwrap();
        let output_nodes = NonZeroUsize::new(1).unwrap();

        let mut layers: Vec<Layer> = Vec::with_capacity(3);
        layers.push(Layer::new(input_nodes, nodes_hidden_layer_1).unwrap());
        layers.push(Layer::new(nodes_hidden_layer_1, nodes_hidden_layer_2).unwrap());
        layers.push(Layer::new(nodes_hidden_layer_2, output_nodes).unwrap());

        let expected_layers: Vec<Layer> = layers.clone();

        let neural_network_result: Result<NeuralNetwork> = NeuralNetwork::new(layers);
        assert!(neural_network_result.is_ok());

        let neural_network: NeuralNetwork = neural_network_result.unwrap();
        assert_eq!(neural_network.get_layers(), expected_layers.as_slice());
    }

    // endregion

    // region AI

    /// Test predicting an output of a neural network for valid input data.
    #[test]
    fn predict_valid_input() {
        let one = NonZeroUsize::new(1).unwrap();
        let input_nodes = NonZeroUsize::new(3).unwrap();
        let nodes_hidden_layer_1 = NonZeroUsize::new(5).unwrap();
        let nodes_hidden_layer_2 = NonZeroUsize::new(4).unwrap();
        let output_nodes = NonZeroUsize::new(2).unwrap();

        let mut layers: Vec<Layer> = Vec::with_capacity(3);
        layers.push(Layer::new(input_nodes, nodes_hidden_layer_1).unwrap());
        layers.push(Layer::new(nodes_hidden_layer_1, nodes_hidden_layer_2).unwrap());
        layers.push(Layer::new(nodes_hidden_layer_2, output_nodes).unwrap());

        let neural_network: NeuralNetwork = NeuralNetwork::new(layers).unwrap();

        let input: Matrix<f64> = Matrix::from_slice(input_nodes, one, &[1.0, 1.1, 1.2]).unwrap();
        let prediction_result: Result<Matrix<f64>> = neural_network.predict(input);
        assert!(prediction_result.is_ok());

        let prediction: Matrix<f64> = prediction_result.unwrap();
        assert_eq!(prediction.get_number_of_rows(), output_nodes.get());
        assert_eq!(prediction.get_number_of_columns(), 1);
        for element in prediction.as_slice() {
            assert!(*element >= 0.0);
            assert!(*element <= 1.0);
        }
    }

    /// Test predicting an output of a neural network if the input matrix has too many columns.
    #[test]
    fn predict_too_many_input_columns() {
        let input_nodes = NonZeroUsize::new(3).unwrap();
        let nodes_hidden_layer_1 = NonZeroUsize::new(5).unwrap();
        let nodes_hidden_layer_2 = NonZeroUsize::new(4).unwrap();
        let output_nodes = NonZeroUsize::new(2).unwrap();

        let mut layers: Vec<Layer> = Vec::with_capacity(3);
        layers.push(Layer::new(input_nodes, nodes_hidden_layer_1).unwrap());
        layers.push(Layer::new(nodes_hidden_layer_1, nodes_hidden_layer_2).unwrap());
        layers.push(Layer::new(nodes_hidden_layer_2, output_nodes).unwrap());

        let neural_network: NeuralNetwork = NeuralNetwork::new(layers).unwrap();

        let input: Matrix<f64> = Matrix::new(input_nodes, output_nodes, 1.0).unwrap();
        let prediction_result: Result<Matrix<f64>> = neural_network.predict(input);

        assert!(
            matches!(prediction_result, Err(Error::DimensionMismatch)),
            "Expected error Error::DimensionMismatch not satisfied."
        );
    }

    /// Test predicting an output of a neural network if the input matrix has the wrong number of
    /// rows.
    #[test]
    fn predict_wrong_number_of_input_rows() {
        let one = NonZeroUsize::new(1).unwrap();
        let input_nodes = NonZeroUsize::new(3).unwrap();
        let nodes_hidden_layer_1 = NonZeroUsize::new(5).unwrap();
        let nodes_hidden_layer_2 = NonZeroUsize::new(4).unwrap();
        let output_nodes = NonZeroUsize::new(2).unwrap();

        let mut layers: Vec<Layer> = Vec::with_capacity(3);
        layers.push(Layer::new(input_nodes, nodes_hidden_layer_1).unwrap());
        layers.push(Layer::new(nodes_hidden_layer_1, nodes_hidden_layer_2).unwrap());
        layers.push(Layer::new(nodes_hidden_layer_2, output_nodes).unwrap());

        let neural_network: NeuralNetwork = NeuralNetwork::new(layers).unwrap();

        let input: Matrix<f64> = Matrix::new(output_nodes, one, 1.0).unwrap();
        let prediction_result: Result<Matrix<f64>> = neural_network.predict(input);

        assert!(
            matches!(prediction_result, Err(Error::DimensionMismatch)),
            "Expected error Error::DimensionMismatch not satisfied."
        );
    }

    // endregion
}
