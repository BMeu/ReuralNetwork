// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! An example usage of the matrix library.

use std::num::NonZeroUsize;

use reural_network::NeuralNetwork;
use reural_network::NeuralNetworkBuilder;

/// The main function.
fn main() {
    let neural_network: NeuralNetwork = NeuralNetworkBuilder::new(NonZeroUsize::new(3).unwrap())
        .add_hidden_layer(NonZeroUsize::new(7).unwrap())
        .add_output_layer(NonZeroUsize::new(10).unwrap())
        .unwrap();

    println!("{:?}", neural_network);
}
