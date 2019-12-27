// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! A simple neural network implementation.

pub use self::error::Error;
pub use self::error::Result;
pub use self::matrix::Matrix;

mod error;
mod matrix;
mod matrix_macros;
