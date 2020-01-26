// Copyright 2020 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Definition and implementation of the neural network.

use crate::Layer;

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
    pub fn new(layers: Vec<Layer>) -> NeuralNetwork {
        NeuralNetwork { layers }
    }

    // endregion
}

#[cfg(test)]
mod tests {

    use super::*;

    use std::num::NonZeroUsize;

    use crate::Layer;

    // region Initialization

    /// Test creating a new neural network.
    #[test]
    fn new() {
        let input_nodes = NonZeroUsize::new(3).unwrap();
        let nodes_hidden_layer_1 = NonZeroUsize::new(5).unwrap();
        let nodes_hidden_layer_2 = NonZeroUsize::new(2).unwrap();
        let output_nodes = NonZeroUsize::new(1).unwrap();

        let mut layers: Vec<Layer> = Vec::with_capacity(3);
        layers.push(Layer::new(input_nodes, nodes_hidden_layer_1).unwrap());
        layers.push(Layer::new(nodes_hidden_layer_1, nodes_hidden_layer_2).unwrap());
        layers.push(Layer::new(nodes_hidden_layer_2, output_nodes).unwrap());

        let neural_network = NeuralNetwork::new(layers);
        assert_eq!(neural_network.layers.len(), 3);
    }

    // endregion
}
