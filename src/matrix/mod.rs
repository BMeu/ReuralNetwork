// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! A simple and naive implementation of mathematical matrices.

pub use self::definition::Matrix;

mod binary_operators_element_wise;
mod binary_operators_scalar;
mod definition;
mod macros;
