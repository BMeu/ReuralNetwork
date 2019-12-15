// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Error handling.

use std::error;
use std::fmt;
use std::result;

/// A specialized `Result` type for Reural Network.
pub type Result<T> = result::Result<T, Error>;

/// A wrapper type for all errors caused by this crate.
#[derive(Debug)]
pub enum Error {
    /// Errors caused if the dimensions of a matrix are invalid.
    MatrixDimension(String),
}

impl fmt::Display for Error {
    /// Format this error using the given formatter.
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::MatrixDimension(ref error) => write!(formatter, "{}", error),
        }
    }
}

impl error::Error for Error {
    /// The underlying source of this error, if any.
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Error::MatrixDimension(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as ErrorTrait;

    use super::*;

    #[test]
    fn fmt_matrix_dimension() {
        let message: &str = "Matrix Dimension Failure";
        let error = Error::MatrixDimension(String::from(message));
        assert_eq!(format!("{}", error), message);
    }

    #[test]
    fn source_matrix_dimension() {
        let message: &str = "Matrix Dimension Failure";
        let error = Error::MatrixDimension(String::from(message));
        assert!(
            error.source().is_none(),
            "Matrix Dimension errors do not have a cause."
        );
    }
}
