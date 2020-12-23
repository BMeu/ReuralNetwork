// Copyright 2020 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Definition and implementation of the builder for creating neural networks.

use std::num::NonZeroUsize;

use crate::Layer;
use crate::NeuralNetwork;
use crate::Result;

/// A builder for creating neural networks.
#[derive(Debug)]
pub struct NeuralNetworkBuilder {
    /// The number of input nodes of the neural network that will be built.
    input_nodes: NonZeroUsize,

    /// For each hidden layer in the neural network, the number of its input nodes.
    hidden_layer_nodes: Vec<NonZeroUsize>,
}

impl NeuralNetworkBuilder {
    /// Start building a new neural network with the given number of input nodes.
    ///
    /// To finish the building process and create the actual neural network, call
    /// [`add_output_layer`].
    ///
    /// [`add_output_layer`]: #method.add_output_layer
    pub fn new(input_nodes: NonZeroUsize) -> Self {
        Self {
            input_nodes,
            hidden_layer_nodes: Vec::new(),
        }
    }

    /// Add a hidden layer with the given number of `nodes` to the neural network.
    ///
    /// The order in which the hidden layers are inserted will be their order in the neural network
    /// once it is built.
    pub fn add_hidden_layer(&'_ mut self, nodes: NonZeroUsize) -> &'_ mut Self {
        self.hidden_layer_nodes.push(nodes);

        self
    }

    // TODO: Describe failures.
    /// Add an output layer with the given number of nodes to the neural network, then initialize
    /// the neural network with the parameters that have been set so far and return it.
    ///
    /// # Undefined Behaviour
    ///
    /// If the number of hidden layers is greater than or equal to [`::std::usize::MAX - 1`], the
    /// behaviour will be undefined.
    ///
    /// [`::std::usize::MAX - 1`]: https://doc.rust-lang.org/stable/std/usize/constant.MAX.html
    pub fn add_output_layer(&self, nodes: NonZeroUsize) -> Result<NeuralNetwork> {
        // Create a vector of all nodes so we can just iterate over all of them.
        // If self.hidden_layer_nodes.len() >= usize::MAX - 1, the addition will silently overflow.
        let number_of_nodes: usize = self.hidden_layer_nodes.len() + 2;
        let mut layer_nodes: Vec<NonZeroUsize> = Vec::with_capacity(number_of_nodes);
        layer_nodes.push(self.input_nodes);
        layer_nodes.append(&mut self.hidden_layer_nodes.clone());
        layer_nodes.push(nodes);

        // Create a copy of the vector, then move ahead to the second item in one of the vectors.
        // We can then just zip those two together and will get a pair of numbers: the first one
        // will be the number of input nodes of a layer and the second one the number of output
        // nodes.
        // E.g.: the vector [1, 4, 2, 7] will result in the pairs [(1, 4), (4, 2), (2, 7)].
        let input_iter: Vec<NonZeroUsize> = layer_nodes.clone();
        let output_iter = layer_nodes.iter().skip(1);

        // Create the layers as described above.
        let mut layers: Vec<Layer> = Vec::with_capacity(layer_nodes.len() - 1);
        for (input_nodes, output_nodes) in input_iter.iter().zip(output_iter) {
            layers.push(Layer::new(*input_nodes, *output_nodes)?)
        }

        // Create and return the actual neural network.
        NeuralNetwork::new(layers)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use std::num::NonZeroUsize;

    use crate::NeuralNetwork;
    use crate::Result;

    /// Test creating a new neural network builder.
    #[test]
    fn new() {
        let input_nodes = NonZeroUsize::new(5).unwrap();
        let builder = NeuralNetworkBuilder::new(input_nodes);

        assert_eq!(builder.input_nodes, input_nodes);
        assert!(builder.hidden_layer_nodes.is_empty())
    }

    /// Test adding hidden layers to the neural network.
    #[test]
    fn add_hidden_layer() {
        let input_nodes = NonZeroUsize::new(5).unwrap();
        let mut builder = NeuralNetworkBuilder::new(input_nodes);

        let nodes_1 = NonZeroUsize::new(7).unwrap();
        builder.add_hidden_layer(nodes_1);
        assert_eq!(builder.input_nodes, input_nodes);
        assert_eq!(builder.hidden_layer_nodes.as_slice(), &[nodes_1]);

        let nodes_2 = NonZeroUsize::new(3).unwrap();
        builder.add_hidden_layer(nodes_2);
        assert_eq!(builder.input_nodes, input_nodes);
        assert_eq!(builder.hidden_layer_nodes.as_slice(), &[nodes_1, nodes_2]);
    }

    /// Test adding an output layer to the neural network and getting a built network.
    #[test]
    fn add_output_layer_success() {
        let input_nodes = NonZeroUsize::new(5).unwrap();
        let nodes_1 = NonZeroUsize::new(7).unwrap();
        let nodes_2 = NonZeroUsize::new(3).unwrap();
        let output_nodes = NonZeroUsize::new(2).unwrap();

        let mut builder = NeuralNetworkBuilder::new(input_nodes);
        builder.add_hidden_layer(nodes_1);
        builder.add_hidden_layer(nodes_2);
        let network_result: Result<NeuralNetwork> = builder.add_output_layer(output_nodes);
        assert!(network_result.is_ok());

        let network: NeuralNetwork = network_result.unwrap();
        let layers: &[Layer] = network.get_layers();
        assert_eq!(layers.len(), 3);

        // Input Layer.
        assert_eq!(layers[0].get_number_of_input_nodes(), input_nodes.get());
        assert_eq!(layers[0].get_number_of_output_nodes(), nodes_1.get());

        // Hidden Layer.
        assert_eq!(layers[1].get_number_of_input_nodes(), nodes_1.get());
        assert_eq!(layers[1].get_number_of_output_nodes(), nodes_2.get());

        // Output Layer.
        assert_eq!(layers[2].get_number_of_input_nodes(), nodes_2.get());
        assert_eq!(layers[2].get_number_of_output_nodes(), output_nodes.get());
    }
}
