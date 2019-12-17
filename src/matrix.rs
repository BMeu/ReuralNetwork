// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! A simple matrix library.

use crate::Error;
use crate::Result;

/// A matrix is a 2-dimensional structure with specific dimensions that can hold data of any type.
///
/// # Example
///
/// A `2x3` matrix of type `f64` could look like this:
///
/// ```text
/// [0.25  1.33 -0.1]
/// [1.0  -2.73  1.2]
/// ```
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
    /// Get the index in the 1-dimensional data vector for the element in row `row` and column
    /// `column`.
    ///
    /// This method does not check if the row and column parameters have valid values. If they are
    /// greater than or equal to the dimensions of the matrix, the resulting index will be out of
    /// bounds.
    ///
    /// This method should only be used if it is guaranteed by the caller that the row and column
    /// are within the dimensions of the matrix.
    fn get_index_unchecked(&self, row: usize, column: usize) -> usize {
        self.columns * row + column
    }

    /// Get the data of the matrix as a 1-dimensional slice.
    ///
    /// For a matrix with `m` rows and `n` columns, the first row of the matrix will become the
    /// first `m` elements in the slice, the second row will become the second `m` elements and so
    /// on.
    ///
    /// # Example
    ///
    /// The following matrix
    ///
    /// ```text
    /// [0 1 2]
    /// [3 4 5]
    /// ```
    ///
    /// will result in the slice:
    ///
    /// ```
    /// let data = [0, 1, 2, 3, 4, 5];
    /// ```
    ///
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
}

impl<T> Matrix<T>
where
    T: Copy,
{
    /// Create a new matrix with the given dimensions and the given default value in all elements.
    ///
    /// The number of rows or the number of columns must be greater than zero and their product must
    /// not exceed the maximum `usize` value, `::std::usize::MAX`.
    ///
    /// # Example
    ///
    /// A `2x3` matrix with a default value of `0.25` for all elements can be created with the
    /// following line of code:
    ///
    /// ```
    /// use reural_network::Matrix;
    /// let matrix: Matrix<f64> = Matrix::new(2, 3, 0.25).unwrap();
    /// ```
    ///"
    pub fn new(rows: usize, columns: usize, default: T) -> Result<Matrix<T>> {
        // The dimensional values must not be 0.
        if rows == 0 || columns == 0 {
            return Err(Error::MatrixDimension(
                "Matrix dimensions must be > 0.".to_owned(),
            ));
        }

        // The size of the data vector cannot be larger than the maximum usize.
        let length: usize = match rows.checked_mul(columns) {
            Some(length) => length,
            None => {
                return Err(Error::MatrixDimension(
                    "The product of rows and columns must not exceed std::usize::MAX.".to_owned(),
                ))
            }
        };

        // Create the data structure and initialize it with the default value.
        let mut data: Vec<T> = Vec::with_capacity(length);
        data.resize(length, default);

        // Return the matrix.
        Ok(Matrix {
            rows,
            columns,
            data,
        })
    }

    /// Convert a slice into a matrix of the given dimensions.
    ///
    /// For a matrix with `m` rows and `n` columns, the first `m` elements in the slice will become
    /// the first row in the matrix, the second `m` elements will become the second row and so on.
    ///
    /// The number of rows or the number of columns must be greater than zero and their product must
    /// not exceed the maximum `usize` value, `::std::usize::MAX`. Furthermore, the product must be
    /// equal to the length of the given data slice.
    ///
    /// # Example
    ///
    /// A `2x3` matrix can be created from a slice of length `6` with the following lines of code:
    ///
    /// ```
    /// use reural_network::Matrix;
    /// let data: [i32; 6] = [0, 1, 2, 3, 4, 5];
    /// let matrix: Matrix<i32> = Matrix::from_slice(2, 3, &data).unwrap();
    /// ```
    ///
    /// This will result in the following matrix:
    ///
    /// ```text
    /// [0 1 2]
    /// [3 4 5]
    /// ```
    ///
    pub fn from_slice(rows: usize, columns: usize, data: &[T]) -> Result<Matrix<T>> {
        // The dimensional values must not be 0.
        if rows == 0 || columns == 0 {
            return Err(Error::MatrixDimension(
                "Matrix dimensions must be > 0.".to_owned(),
            ));
        }

        // The size of the data vector cannot be larger than the maximum usize.
        let length: usize = match rows.checked_mul(columns) {
            Some(length) => length,
            None => {
                return Err(Error::MatrixDimension(
                    "The product of rows and columns must not exceed std::usize::MAX.".to_owned(),
                ))
            }
        };

        // Check that the length of the data slice matches the dimensions of the matrix.
        if length != data.len() {
            return Err(Error::MatrixDimension(
                "The length of the data slice must be equal to the product of rows and columns."
                    .to_owned(),
            ));
        }

        // Return the matrix.
        Ok(Matrix {
            rows,
            columns,
            data: data.to_vec(),
        })
    }

    /// Map each value in the matrix to a new value as given by the closure `mapping`.
    ///
    /// # Example
    ///
    /// Convert a matrix of temperatures in °C to °F:
    ///
    /// ```
    /// use reural_network::Matrix;
    /// let temperatures: [usize; 6] = [0, 10, 25, 50, 75, 100];
    /// let mut matrix: Matrix<usize> = Matrix::from_slice(2, 3, &temperatures).unwrap();
    /// // Convert Celsius to Fahrenheit.
    /// matrix.map(|celsius| (celsius * 9 / 5) + 32);
    /// ```
    pub fn map<F>(&mut self, mapping: F)
    where
        F: Fn(T) -> T,
    {
        for row in 0..self.rows {
            for column in 0..self.columns {
                let index: usize = self.get_index_unchecked(row, column);

                // The data vector has been created with the required size in `new()`, so we can
                // directly access the index without having to check it for existence.
                self.data[index] = mapping(self.data[index]);
            }
        }
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
        let matrix_result: Result<Matrix<usize>> = Matrix::new(rows, columns, 0);

        assert!(matrix_result.is_ok());

        let matrix: Matrix<usize> = matrix_result.unwrap();
        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.columns, columns);
        assert_eq!(matrix.data, [0usize; 15].to_vec());

        // Invalid dimensions.
        let rows: usize = 0;
        let matrix_result: Result<Matrix<usize>> = Matrix::new(rows, columns, 0);

        assert!(matrix_result.is_err());

        let error = matrix_result.unwrap_err();
        match error {
            Error::MatrixDimension(ref description) => {
                assert_eq!(description, "Matrix dimensions must be > 0.")
            }
        }

        // Too large dimensions.
        let rows: usize = ::std::usize::MAX;
        let columns: usize = 2;
        let matrix_result: Result<Matrix<usize>> = Matrix::new(rows, columns, 0);

        assert!(matrix_result.is_err());
        let error = matrix_result.unwrap_err();
        match error {
            Error::MatrixDimension(ref description) => assert_eq!(
                description,
                "The product of rows and columns must not exceed std::usize::MAX."
            ),
        }
    }

    /// Test creating a new matrix from a slice.
    #[test]
    fn from_slice() {
        // Valid dimensions.
        let rows: usize = 5;
        let columns: usize = 3;
        let data: [usize; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let matrix_result: Result<Matrix<usize>> = Matrix::from_slice(rows, columns, &data);

        assert!(matrix_result.is_ok());

        let matrix: Matrix<usize> = matrix_result.unwrap();
        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.columns, columns);
        assert_eq!(matrix.data, data.to_vec());

        // Invalid dimensions.
        let rows: usize = 0;
        let matrix_result: Result<Matrix<usize>> = Matrix::from_slice(rows, columns, &data);

        assert!(matrix_result.is_err());

        let error = matrix_result.unwrap_err();
        match error {
            Error::MatrixDimension(ref description) => {
                assert_eq!(description, "Matrix dimensions must be > 0.")
            }
        }

        // Too large dimensions.
        let rows: usize = ::std::usize::MAX;
        let columns: usize = 2;
        let matrix_result: Result<Matrix<usize>> = Matrix::from_slice(rows, columns, &data);

        assert!(matrix_result.is_err());
        let error = matrix_result.unwrap_err();
        match error {
            Error::MatrixDimension(ref description) => assert_eq!(
                description,
                "The product of rows and columns must not exceed std::usize::MAX."
            ),
        }

        // Dimension mismatch with data vector.
        let rows: usize = 5;
        let columns: usize = 3;
        let data: [usize; 5] = [0, 1, 2, 3, 4];
        let matrix_result: Result<Matrix<usize>> = Matrix::from_slice(rows, columns, &data);

        assert!(matrix_result.is_err());
        let error = matrix_result.unwrap_err();
        match error {
            Error::MatrixDimension(ref description) => assert_eq!(
                description,
                "The length of the data slice must be equal to the product of rows and columns."
            ),
        }
    }

    /// Test getting the unchecked index for given rows and columns.
    #[test]
    fn get_index_unchecked() {
        let rows: usize = 10;
        let columns: usize = 10;
        let matrix: Matrix<usize> = Matrix::new(rows, columns, 0).unwrap();

        // (0, 0) => 0
        assert_eq!(matrix.get_index_unchecked(0, 0), 0);

        // (0, 1) => 1
        assert_eq!(matrix.get_index_unchecked(0, 1), 1);

        // (1, 0) => 10
        assert_eq!(matrix.get_index_unchecked(1, 0), 10);

        // (3, 7) => 36
        assert_eq!(matrix.get_index_unchecked(1, 0), 10);

        // (9, 9) => 99
        assert_eq!(matrix.get_index_unchecked(9, 9), 99);

        // (10, 0) => 100 (out of bounds)
        assert_eq!(matrix.get_index_unchecked(10, 0), 100);
    }

    /// Test mapping the data in a matrix.
    #[test]
    fn map() {
        let temperatures = [0, 10, 25, 50, 75, 100];
        let mut temperature: Matrix<usize> = Matrix::from_slice(2, 3, &temperatures).unwrap();

        // Convert Celsius to Fahrenheit.
        temperature.map(|celsius| (celsius * 9 / 5) + 32);

        // Temperature in °F.
        assert_eq!(temperature.data, vec![32, 50, 77, 122, 167, 212]);
    }
}
