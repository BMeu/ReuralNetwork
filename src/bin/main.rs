// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! An example usage of the matrix library.

use reural_network::Matrix;

/// The main function.
fn main() {
    let data: [f64; 6] = [0.25, 1.33, -0.1, 1.0, -2.73, 1.2];
    let matrix: Matrix<f64> = Matrix::from_slice(2, 3, &data).unwrap();

    println!("{}", matrix);
}
