// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Error handling and related stuff.

use std::error;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::result::Result as StdResult;

/// A specialized `Result` type for Reural Network.
pub type Result<T> = StdResult<T, Error>;

/// A wrapper type for all errors caused by this crate.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// If an element is accessed whose coordinates (row and column) are not within the matrix.
    CellOutOfBounds,

    /// If the dimensions of a matrix do not match the dimensions of another matrix or the length of
    /// a slice from which a matrix with specific dimensions is created, this error will be
    /// returned.
    DimensionMismatch,

    /// If the dimensions of a matrix exceed the maximum allowed value, this error will be returned.
    DimensionsTooLarge,
}

impl Display for Error {
    /// Format this error using the given formatter.
    fn fmt(&self, formatter: &mut Formatter) -> FmtResult {
        match *self {
            Error::DimensionMismatch => write!(
                formatter,
                "The dimensions of the matrices must be the same or the length of the slice must match the dimensions of the matrix."
            ),
            Error::DimensionsTooLarge => write!(
                formatter,
                "The product of rows and columns must not exceed the maximum usize value, ::std::usize::MAX."
            ),
            Error::CellOutOfBounds => write!(
                formatter,
                "The cell is not part of the matrix."
            ),
        }
    }
}

impl error::Error for Error {
    /// The underlying source of this error, if any.
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use super::*;

    /// Test debug formatting a `CellOutOfBounds` error.
    #[test]
    fn debug_cell_out_of_bounds() {
        let error = Error::CellOutOfBounds;
        assert_eq!(format!("{:?}", error), "CellOutOfBounds");
    }

    /// Test debug formatting a `DimensionMismatch` error.
    #[test]
    fn debug_dimension_mismatch() {
        let error = Error::DimensionMismatch;
        assert_eq!(format!("{:?}", error), "DimensionMismatch");
    }

    /// Test debug formatting a `DimensionsTooLarge` error.
    #[test]
    fn debug_dimensions_too_large() {
        let error = Error::DimensionsTooLarge;
        assert_eq!(format!("{:?}", error), "DimensionsTooLarge");
    }

    /// Test formatting a `CellOutOfBounds` error.
    #[test]
    fn fmt_cell_out_of_bounds() {
        let error = Error::CellOutOfBounds;
        assert_eq!(format!("{}", error), "The cell is not part of the matrix.");
    }

    /// Test formatting a `DimensionMismatch` error.
    #[test]
    fn fmt_dimension_mismatch() {
        let error = Error::DimensionMismatch;
        assert_eq!(
            format!("{}", error),
            "The dimensions of the matrices must be the same or the length of the slice must match the dimensions of the matrix."
        );
    }

    /// Test formatting a `DimensionsTooLarge` error.
    #[test]
    fn fmt_dimensions_too_large() {
        let error = Error::DimensionsTooLarge;
        assert_eq!(
            format!("{}", error),
            "The product of rows and columns must not exceed the maximum usize value, ::std::usize::MAX."
        );
    }

    /// Test getting the source of a `CellOutOfBounds` error.
    #[test]
    fn source_cell_out_of_bounds() {
        let error = Error::CellOutOfBounds;
        assert!(error.source().is_none());
    }

    /// Test getting the source of a `DimensionsMismatch` error.
    #[test]
    fn source_dimension_mismatch() {
        let error = Error::DimensionMismatch;
        assert!(error.source().is_none());
    }

    /// Test getting the source of a `DimensionsTooLarge` error.
    #[test]
    fn source_dimensions_too_large() {
        let error = Error::DimensionsTooLarge;
        assert!(error.source().is_none());
    }
}
