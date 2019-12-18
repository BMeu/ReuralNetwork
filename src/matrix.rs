// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! A simple matrix library.

use std::fmt::Display;
use std::fmt::Formatter;
use std::num::NonZeroUsize;
use std::ops::Add;

use crate::Error;
use crate::Result;
use std::cmp::max;

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

    /// Get the number of rows in the matrix.
    pub fn get_rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns in the matrix.
    pub fn get_columns(&self) -> usize {
        self.columns
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
    /// use std::num::NonZeroUsize;
    /// use reural_network::Matrix;
    ///
    /// let rows = NonZeroUsize::new(2).unwrap();
    /// let columns = NonZeroUsize::new(3).unwrap();
    /// let matrix: Matrix<f64> = Matrix::new(rows, columns, 0.25).unwrap();
    /// ```
    ///"
    pub fn new(rows: NonZeroUsize, columns: NonZeroUsize, default: T) -> Result<Matrix<T>> {
        let num_rows: usize = rows.get();
        let num_columns: usize = columns.get();

        // The size of the data vector cannot be larger than the maximum usize.
        let length: usize = match num_rows.checked_mul(num_columns) {
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
            rows: num_rows,
            columns: num_columns,
            data,
        })
    }

    /// Get the value in the given `row` and `column`.
    ///
    /// If the `row` or `column` value is larger than the number of rows or columns in the matrix,
    /// respectively, an error will returned.
    ///
    /// If it can be guaranteed that the `row` and `column` values do not exceed the dimensions,
    /// you can also use `get_unchecked()`.
    pub fn get(&self, row: usize, column: usize) -> Result<T> {
        if row >= self.rows || column >= self.columns {
            return Err(Error::MatrixDimension("".to_owned()));
        }

        Ok(self.get_unchecked(row, column))
    }

    /// Get the value in the given `row` and `column`.
    ///
    /// This method does not check if the row and column parameters have valid values. If they are
    /// greater than or equal to the dimensions of the matrix, the resulting index will be out of
    /// bounds.
    ///
    /// This method should only be used if it is guaranteed by the caller that the row and column
    /// are within the dimensions of the matrix. If this cannot be guaranteed, use `get()` instead.
    pub fn get_unchecked(&self, row: usize, column: usize) -> T {
        self.data[self.get_index_unchecked(row, column)]
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
    /// use std::num::NonZeroUsize;
    /// use reural_network::Matrix;
    ///
    /// let rows = NonZeroUsize::new(2).unwrap();
    /// let columns = NonZeroUsize::new(3).unwrap();
    /// let matrix: Matrix<i32> = Matrix::from_slice(rows, columns, &[0, 1, 2, 3, 4, 5]).unwrap();
    /// ```
    ///
    /// This will result in the following matrix:
    ///
    /// ```text
    /// [0 1 2]
    /// [3 4 5]
    /// ```
    ///
    pub fn from_slice(rows: NonZeroUsize, columns: NonZeroUsize, data: &[T]) -> Result<Matrix<T>> {
        let num_rows: usize = rows.get();
        let num_columns: usize = columns.get();

        // The size of the data vector cannot be larger than the maximum usize.
        let length: usize = match num_rows.checked_mul(num_columns) {
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
            rows: num_rows,
            columns: num_columns,
            data: data.to_vec(),
        })
    }

    /// Map each value in the matrix to a new value as given by the closure `mapping`.
    ///
    /// The `mapping` function needs three parameters:
    ///
    /// 1. The value of the current element.
    /// 2. The row of the current element.
    /// 3. The column of the current element.
    ///
    /// It must return the new value of the current element.
    ///
    /// # Example
    ///
    /// Convert a matrix of temperatures in °C to °F:
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use reural_network::Matrix;
    ///
    /// // Convert Celsius to Fahrenheit.
    /// let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
    /// let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
    /// let temperatures: [usize; 6] = [0, 10, 25, 50, 75, 100];
    /// let mut matrix: Matrix<usize> = Matrix::from_slice(rows, columns, &temperatures).unwrap();
    /// matrix.map(|celsius, _row, _column| (celsius * 9 / 5) + 32);
    /// ```
    pub fn map<F>(&mut self, mapping: F)
    where
        F: Fn(T, usize, usize) -> T,
    {
        for row in 0..self.rows {
            for column in 0..self.columns {
                let index: usize = self.get_index_unchecked(row, column);

                // The data vector has been created with the required size in `new()`, so we can
                // directly access the index without having to check it for existence.
                self.data[index] = mapping(self.data[index], row, column);
            }
        }
    }

    /// Transpose this matrix.
    ///
    /// # Example
    ///
    /// A `2x3` matrix
    ///
    /// ```text
    /// [0 1 2]
    /// [3 4 5]
    /// ```
    ///
    /// will become a `3x2` matrix:
    ///
    /// ```text
    /// [0 3]
    /// [1 4]
    /// [2 5]
    /// ```
    ///
    /// In code, this will look as follows:
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use reural_network::Matrix;
    ///
    /// let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
    /// let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
    /// let matrix: Matrix<usize> = Matrix::from_slice(rows, columns, &[0, 1, 2, 3, 4, 5]).unwrap();
    /// let transposed: Matrix<usize> = matrix.transpose();
    ///
    /// assert_eq!(transposed.get_rows(), 3);
    /// assert_eq!(transposed.get_columns(), 2);
    /// assert_eq!(transposed.as_slice(), &[0, 3, 1, 4, 2, 5]);
    /// ```
    pub fn transpose(&self) -> Matrix<T> {
        // The rows and columns are switched in the transposed matrix.
        let rows: usize = self.columns;
        let columns: usize = self.rows;

        // Allocate the required memory at once. This is faster than having to resize the vector
        // every few insertions.
        let length: usize = rows * columns;
        let mut data: Vec<T> = Vec::with_capacity(length);
        for index in 0..length {
            // Basically, iterate over the new data vector (which is still empty).
            // At every index of the new vector, find the corresponding value from the original
            // matrix based on the index.

            // Get row and column for this index in the transposed matrix.
            let row: usize = index / columns;
            let column: usize = index % columns;

            // Rows and columns are switched in the transposed matrix, so consider this when getting
            // the index for the original data.
            let value: T = self.get_unchecked(column, row);
            data.push(value);
        }

        Matrix {
            rows: self.columns,
            columns: self.rows,
            data,
        }
    }
}

impl<T> Display for Matrix<T>
where
    T: Display,
{
    /// Format the data in the matrix.
    fn fmt(&self, formatter: &mut Formatter) -> ::std::fmt::Result {
        // Align all columns, but each column may have a different alignment. Thus, first iterate
        // over the columns, then the rows, to get the width of each column from all values in the
        // column.
        let mut column_widths: Vec<usize> = Vec::with_capacity(self.columns);
        for column in 0..self.columns {
            let mut max_width: usize = 0;

            // Get all values in the column.
            for row in 0..self.rows {
                // Do not use self.get_unchecked() here as this requires Copy for T.
                let value: String = format!("{}", self.data[self.get_index_unchecked(row, column)]);
                let width: usize = value.len();
                max_width = max(max_width, width);
            }

            column_widths.push(max_width);
        }

        // Now, go through each row and format the value with the corresponding columns width.
        let mut rows: Vec<String> = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let mut row_values: Vec<String> = Vec::with_capacity(self.columns);

            for (column, width) in column_widths.iter().enumerate() {
                let value: String = format!(
                    // Left-align all values.
                    "{:<width$}",
                    // Do not use self.get_unchecked() here as this requires Copy for T.
                    self.data[self.get_index_unchecked(row, column)],
                    width = width
                );

                row_values.push(value);
            }

            // Concatenate all aligned values in the row with three spaces. Surround the values with
            // square brackets.
            rows.push(format!("[{}]", row_values.join("   ")));
        }

        // Concatenate all rows with a new line.
        write!(formatter, "{}", rows.join("\n"))
    }
}

impl<'a> Add<f64> for &'a Matrix<f64> {
    type Output = Matrix<f64>;

    /// Add the scalar `value` to each element in the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use reural_network::Matrix;
    ///
    /// let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
    /// let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
    /// let data: [f64; 6] = [0.25, 1.33, -0.1, 1.0, -2.73, 1.2];
    /// let matrix: Matrix<f64> = Matrix::from_slice(rows, columns, &data).unwrap();
    ///
    /// let result: Matrix<f64> = &matrix + 1f64;
    /// assert_eq!(result.as_slice(), [1.25, 2.33, 0.9, 2.0, -1.73, 2.2]);
    /// ```
    ///
    fn add(self, value: f64) -> Self::Output {
        let mut result: Matrix<f64> = Matrix {
            rows: self.rows,
            columns: self.columns,
            data: self.data.clone(),
        };

        result.map(|element, _row, _column| element + value);

        result
    }
}

impl<'a, 'b> Add<&'b Matrix<f64>> for &'a Matrix<f64> {
    type Output = Result<Matrix<f64>>;

    /// Element-wise add the `other` matrix to this matrix.
    ///
    /// The dimensions of both matrices must match, otherwise an error will be returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use reural_network::Matrix;
    /// use reural_network::Result;
    ///
    /// let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
    /// let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
    /// let data: [f64; 6] = [0.25, 1.33, -0.1, 1.0, -2.73, 1.2];
    /// let matrix: Matrix<f64> = Matrix::from_slice(rows, columns, &data).unwrap();
    /// let other: Matrix<f64> = Matrix::from_slice(rows, columns, &data).unwrap();
    ///
    /// let result: Result<Matrix<f64>> = &matrix + &other;
    /// assert!(result.is_ok());
    /// assert_eq!(result.unwrap().as_slice(), [0.5, 2.66, -0.2, 2.0, -5.46, 2.4]);
    /// ```
    ///
    fn add(self, other: &'b Matrix<f64>) -> Self::Output {
        if self.rows != other.get_rows() || self.columns != other.get_columns() {
            return Err(Error::MatrixDimension("".to_owned()));
        }

        let mut result: Matrix<f64> = Matrix {
            rows: self.rows,
            columns: self.columns,
            data: self.data.clone(),
        };

        result.map(|element, row, column| element + other.get_unchecked(row, column));

        Ok(result)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    /// Test creating a new matrix.
    #[test]
    fn new() {
        // Valid dimensions.
        let rows: NonZeroUsize = NonZeroUsize::new(5).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let matrix_result: Result<Matrix<usize>> = Matrix::new(rows, columns, 0);

        assert!(matrix_result.is_ok());

        let matrix: Matrix<usize> = matrix_result.unwrap();
        assert_eq!(matrix.rows, rows.get());
        assert_eq!(matrix.columns, columns.get());
        assert_eq!(matrix.as_slice(), [0usize; 15]);

        // Too large dimensions.
        let rows: NonZeroUsize = NonZeroUsize::new(::std::usize::MAX).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(2).unwrap();
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
        let rows: NonZeroUsize = NonZeroUsize::new(5).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [usize; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let matrix_result: Result<Matrix<usize>> = Matrix::from_slice(rows, columns, &data);

        assert!(matrix_result.is_ok());

        let matrix: Matrix<usize> = matrix_result.unwrap();
        assert_eq!(matrix.rows, rows.get());
        assert_eq!(matrix.columns, columns.get());
        assert_eq!(matrix.as_slice(), data);

        // Too large dimensions.
        let rows: NonZeroUsize = NonZeroUsize::new(::std::usize::MAX).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(2).unwrap();
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
        let rows: NonZeroUsize = NonZeroUsize::new(5).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
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
        let rows: NonZeroUsize = NonZeroUsize::new(10).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(10).unwrap();
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
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let mut temperature: Matrix<usize> =
            Matrix::from_slice(rows, columns, &temperatures).unwrap();

        // Convert Celsius to Fahrenheit.
        temperature.map(|celsius, _row, _column| (celsius * 9 / 5) + 32);

        // Temperature in °F.
        assert_eq!(temperature.as_slice(), [32, 50, 77, 122, 167, 212]);
    }

    /// Test transposing a matrix.
    #[test]
    fn transpose() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [usize; 6] = [0, 1, 2, 3, 4, 5];
        let matrix: Matrix<usize> = Matrix::from_slice(rows, columns, &data).unwrap();
        let transposed: Matrix<usize> = matrix.transpose();

        assert_eq!(transposed.rows, columns.get());
        assert_eq!(transposed.columns, rows.get());
        assert_eq!(transposed.as_slice(), [0, 3, 1, 4, 2, 5]);
    }

    /// Test pretty-printing the matrix.
    #[test]
    fn display() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [f64; 6] = [0.25, 1.33, -0.1, 1.0, -2.73, 1.2];
        let matrix: Matrix<f64> = Matrix::from_slice(rows, columns, &data).unwrap();
        let display = format!("{}", matrix);
        assert_eq!("[0.25   1.33    -0.1]\n[1      -2.73   1.2 ]", display);
    }

    /// Test adding a scalar `f64` value to a matrix.
    #[test]
    fn add_scalar_f64() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [f64; 6] = [0.25, 1.33, -0.1, 1.0, -2.73, 1.2];
        let matrix: Matrix<f64> = Matrix::from_slice(rows, columns, &data).unwrap();

        let result: Matrix<f64> = &matrix + 1f64;
        assert_eq!(result.as_slice(), [1.25, 2.33, 0.9, 2.0, -1.73, 2.2]);
    }

    /// Test adding a matrix to another matrix.
    #[test]
    fn add_matrix_f64() {
        // Matching dimensions.
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [f64; 6] = [0.25, 1.33, -0.1, 1.0, -2.73, 1.2];
        let matrix: Matrix<f64> = Matrix::from_slice(rows, columns, &data).unwrap();
        let other: Matrix<f64> = Matrix::from_slice(rows, columns, &data).unwrap();

        let result: Result<Matrix<f64>> = &matrix + &other;
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().as_slice(),
            [0.5, 2.66, -0.2, 2.0, -5.46, 2.4]
        );

        // Wrong dimensions.
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [f64; 6] = [0.25, 1.33, -0.1, 1.0, -2.73, 1.2];
        let matrix: Matrix<f64> = Matrix::from_slice(rows, columns, &data).unwrap();
        let other: Matrix<f64> = Matrix::from_slice(columns, rows, &data).unwrap();

        let result: Result<Matrix<f64>> = &matrix + &other;
        assert!(result.is_err());
    }

    /// Test getting a value when the row or column are valid.
    #[test]
    fn get_valid_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [u64; 6] = [10, 11, 12, 13, 14, 15];
        let matrix: Matrix<u64> = Matrix::from_slice(rows, columns, &data).unwrap();

        let value: Result<u64> = matrix.get(0, 0);
        assert!(value.is_ok());
        assert_eq!(value.unwrap(), 10);
    }

    /// Test getting a value when the row or column are invalid.
    #[test]
    fn get_invalid_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [u64; 6] = [10, 11, 12, 13, 14, 15];
        let matrix: Matrix<u64> = Matrix::from_slice(rows, columns, &data).unwrap();

        let value: Result<u64> = matrix.get(2, 5);
        assert!(value.is_err());
    }

    /// Test getting a value without checking the row and column when the row or column are valid.
    #[test]
    fn get_unchecked_valid_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [u64; 6] = [10, 11, 12, 13, 14, 15];
        let matrix: Matrix<u64> = Matrix::from_slice(rows, columns, &data).unwrap();

        assert_eq!(matrix.get_unchecked(0, 0), 10);
        assert_eq!(matrix.get_unchecked(0, 1), 11);
        assert_eq!(matrix.get_unchecked(0, 2), 12);
        assert_eq!(matrix.get_unchecked(1, 0), 13);
        assert_eq!(matrix.get_unchecked(1, 1), 14);
        assert_eq!(matrix.get_unchecked(1, 2), 15);
    }

    /// Test getting a value without checking the row and column when the row or column are invalid.
    #[test]
    #[should_panic(expected = "index out of bounds: the len is 6 but the index is 11")]
    fn get_unchecked_invalid_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [u64; 6] = [10, 11, 12, 13, 14, 15];
        let matrix: Matrix<u64> = Matrix::from_slice(rows, columns, &data).unwrap();

        let _: u64 = matrix.get_unchecked(2, 5);
    }
}
