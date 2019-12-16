// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! An example usage of the matrix library.

use reural_network::Matrix;
use reural_network::Result;

/// The main function.
fn main() {
    let matrix: Result<Matrix<usize>> = Matrix::new(2, 2);

    println!("{:?}", matrix);
}
