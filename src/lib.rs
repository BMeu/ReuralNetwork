// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! A simple neural network implementation.

pub use self::error::Error;
pub use self::error::Result;

// TODO: Make the matrix and layer modules private once main.rs doesn't use them anymore.
mod error;
pub mod layer;
mod macros;
pub mod matrix;
