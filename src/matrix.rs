// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! A simple matrix library.

use crate::Error;
use crate::Result;

/// The matrix.
#[derive(Debug)]
pub struct Matrix<T> {
    /// The number of rows.
    rows: usize,

    /// The number of columns.
    columns: usize,

    /// The actual data of the matrix as a 1-dimensional array.
    data: Vec<T>,
}

impl<T> Matrix<T> {
    /// Create a new matrix with the given dimensions.
    ///
    /// This will allocate the memory required to hold all data in the matrix.
    pub fn new(rows: usize, columns: usize) -> Result<Matrix<T>> {
        if rows == 0 || columns == 0 {
            return Err(Error::MatrixDimension(
                "Matrix dimensions must be > 0.".to_owned(),
            ));
        }

        let data: Vec<T> = Vec::with_capacity(rows * columns);

        Ok(Matrix {
            rows,
            columns,
            data,
        })
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    /// Test creating a new matrix.
    #[test]
    fn new() {
        // Valid dimensions.
        let rows: usize = 5;
        let columns: usize = 3;
        let matrix_result: Result<Matrix<usize>> = Matrix::new(rows, columns);

        assert!(matrix_result.is_ok());

        let matrix: Matrix<usize> = matrix_result.unwrap();
        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.columns, columns);
        assert_eq!(matrix.data.capacity(), rows * columns);

        // Invalid dimensions.
        let rows: usize = 0;
        let matrix_result: Result<Matrix<usize>> = Matrix::new(rows, columns);

        assert!(matrix_result.is_err());

        let error = matrix_result.unwrap_err();
        match error {
            Error::MatrixDimension(ref description) => {
                assert_eq!(description, "Matrix dimensions must be > 0.")
            }
            _ => assert!(false, "Wrong error type."),
        }
    }
}
