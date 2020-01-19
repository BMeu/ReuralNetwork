// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Definition and implementation of the matrix struct.

use std::cmp::max;
use std::fmt::Display;
use std::fmt::Formatter;
use std::num::NonZeroUsize;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::BitAnd;
use std::ops::BitAndAssign;
use std::ops::BitOr;
use std::ops::BitOrAssign;
use std::ops::BitXor;
use std::ops::BitXorAssign;
use std::ops::Div;
use std::ops::DivAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Not;
use std::ops::Rem;
use std::ops::RemAssign;
use std::ops::Shl;
use std::ops::ShlAssign;
use std::ops::Shr;
use std::ops::ShrAssign;
use std::ops::Sub;
use std::ops::SubAssign;

use rand::distributions::Uniform;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand::Rng;

use crate::impl_element_wise_binary_operators;
use crate::impl_scalar_assign_operators;
use crate::impl_scalar_binary_operators;
use crate::impl_unary_operators;
use crate::Error;
use crate::Result;

/// A matrix is a 2-dimensional structure with specific dimensions that can hold data of any type.
///
/// Rows and columns of a matrix are zero-indexed, meaning that the top left element of the matrix
/// is in row `0` and column `0`, and the bottom right elememt is in row `rows - 1` and column
/// `columns - 1`, where `rows` and `columns` are the number of rows and columns the matrix has,
/// respectively.
///
/// # Example
///
/// Create a new matrix from a slice of data, add a scalar value to it and transpose the result:
///
/// ```
/// use std::num::NonZeroUsize;
/// use reural_network::matrix::Matrix;
///
/// let rows = NonZeroUsize::new(2).unwrap();
/// let columns = NonZeroUsize::new(3).unwrap();
/// let elements: [f64; 6] = [2.3, 4.0, 3.3, -1.465, 0.0, -42.0];
///
/// // Create a 2x3 matrix.
/// let matrix = Matrix::from_slice(rows, columns, &elements).unwrap();
///
/// // Add 7.3 to each element.
/// let sum = &matrix + 7.3;
///
/// // Transpose the sum to get a 3x2 matrix.
/// let transposed = sum.transpose();
///
/// assert_eq!(transposed.as_slice(), &[9.6, 5.835, 11.3, 7.3, 10.6, -34.7]);
/// ```
///
/// # Size Limits
///
/// A matrix can hold a maximum number of [`::std::usize::MAX`] elements. When creating a new
/// matrix, it is checked if the given number of rows and columns would create a matrix that would
/// exceed this size limit. In this case, the matrix cannot be created.
///
/// # Supported Mathematical Operations
///
/// The following mathematical operations are supported for matrices `Matrix<T>`:
///
/// * Addition[<sup>*</sup>]
///     * Element-wise addition of two matrices with the same dimensions.
///     * Scalar addition of a matrix and a scalar value.
/// * Subtraction[<sup>*</sup>]
///     * Element-wise subtraction of two matrices with the same dimensions.
///     * Scalar subtraction of a scalar value from a matrix.
/// * Multiplication[<sup>*</sup>]
///     * Element-wise multiplication (Hadamard product) of two matrices with the same dimensions.
///     * Scalar multiplication of a scalar value to a matrix.
/// * Division[<sup>*</sup>]
///     * Element-wise division of two matrices with the same dimensions.
///     * Scalar division of a matrix by a scalar value.
/// * Bitwise AND[<sup>*</sup>]
///     * Element-wise bitwise AND of two matrices with the same dimensions.
///     * Scalar bitwise AND of a matrix and a scalar value.
/// * Bitwise OR[<sup>*</sup>]
///     * Element-wise bitwise OR of two matrices with the same dimensions.
///     * Scalar bitwise OR of a matrix and a scalar value.
/// * Bitwise XOR[<sup>*</sup>]
///     * Element-wise bitwise XOR of two matrices with the same dimensions.
///     * Scalar bitwise XOR of a matrix and a scalar value.
/// * Left Shift[<sup>*</sup>]
///     * Element-wise left shift of a matrix by another matrix with the same dimensions.
///     * Scalar left shift of a matrix by a scalar value.
/// * Right Shift[<sup>*</sup>]
///     * Element-wise right shift of a matrix by another matrix with the same dimensions.
///     * Scalar right shift of a matrix by a scalar value.
/// * Negation:[<sup>*</sup>] Negate all elements of a matrix.
/// * Logical Negation:[<sup>*</sup>] Logically negate all elements in a matrix.
/// * Matrix Multiplication: compute the matrix product of two matrices ([`matrix_mul`]).
/// * Transposition: flipping a matrix over its diagonal ([`transpose`]).
/// * Map: change each element in a matrix based on a closure ([`map`]).
///
/// These operations use a naive implementation without any considerations for performance.
///
/// The binary operations are also implemented as assignment operators for scalar values (e.g.
/// `matrix += 4.5`). However, they are not implemented as assignment operators for matrices (e.g.
/// `matrix += other_matrix`) since both matrices must have the same dimensions. If the other matrix
/// did not have the correct dimensions, we could either not perform the operation on `matrix`, or
/// we could panic. Since both of these possibilities can be very surprising to the user, the
/// assignment operators are not implemeneted for matrices. Instead, you will have to resort to
/// solutions like:
///
/// ```
/// # use std::num::NonZeroUsize;
/// # use reural_network::matrix::Matrix;
/// # use reural_network::Result;
/// #
/// # let rows = NonZeroUsize::new(1).unwrap();
/// # let columns = NonZeroUsize::new(1).unwrap();
/// # let matrix = Matrix::from_slice(rows, columns, &[0.5]).unwrap();
/// # let other_matrix = Matrix::from_slice(rows, columns, &[0.5]).unwrap();
/// #
/// let result = matrix + other_matrix;
/// let matrix = result.unwrap();
/// ```
///
/// -----
///
/// <a name="impl-note-operations"><sup>*</sup></a> The operation must be implemeneted for the type
/// `T`.
///
/// [<sup>*</sup>]: #impl-note-operations
/// [`map`]: #method.map
/// [`matrix_mul`]: #method.matrix_mul
/// [`transpose`]: #method.transpose
/// [`::std::usize::MAX`]: https://doc.rust-lang.org/stable/std/usize/constant.MAX.html
#[derive(Debug)]
pub struct Matrix<T> {
    /// The number of rows the matrix has.
    rows: NonZeroUsize,

    /// The number of columns the matrix has.
    columns: NonZeroUsize,

    /// The actual data of the matrix as a 1-dimensional array.
    ///
    /// For a matrix with `m` rows and `n` columns, the first `m` elements in the vector will be the
    /// first row of the matrix, the second `m` elements will be the second row and so on.
    data: Vec<T>,
}

impl<T> Matrix<T> {
    // region Getters

    /// Get the data of the matrix as a 1-dimensional slice.
    ///
    /// For a matrix with `m` rows and `n` columns, the first row of the matrix will become the
    /// first `m` elements in the slice, the second row will become the second `m` elements and so
    /// on.
    ///
    /// # Example
    ///
    /// The matrix
    ///
    /// ```text
    /// [0 1 2]
    /// [3 4 5]
    /// ```
    ///
    /// will produce the following slice:
    ///
    /// ```
    /// let data: &[usize] = &[0, 1, 2, 3, 4, 5];
    /// ```
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Get the number of columns in the matrix.
    pub fn get_columns(&self) -> usize {
        self.columns.get()
    }

    /// Get the index in the 1-dimensional data vector for the element in the given `row` and
    /// `column`.
    ///
    /// This method does not check if the row and column parameters have valid values. If they are
    /// greater than or equal to the dimensions of the matrix, the resulting index will be out of
    /// bounds.
    ///
    /// This method should only be used if it is guaranteed by the caller that the row and column
    /// are within the dimensions of the matrix.
    ///
    /// # Safety
    ///
    /// If the row or column are out of bounds of the matrix, the computed index for the internal
    /// data structure will be out of bounds. Using this index without any further checks to access
    /// data in the data structure will cause a panic.
    ///
    /// # Guarantees
    ///
    /// When iterating over all elements in a matrix in row-major format, the index will be
    /// increasing by exactly 1.
    unsafe fn get_index_unchecked(&self, row: usize, column: usize) -> usize {
        self.columns.get() * row + column
    }

    /// Get the length of the data vector based on the number of rows and columns.
    ///
    /// The product of `rows` and `columns` must not exceed the maximum `usize` value,
    /// [`::std::usize::MAX`]. If this invariant is uphold, the length will be returned. Otherwise,
    /// an [`Error::DimensionsTooLarge`] will be returned.
    ///
    /// If you can guarantee that the resulting length will not exceed the maximum size of the
    /// matrix, you can also use [`get_length_from_rows_and_columns_unchecked`].
    ///
    /// [`::std::usize::MAX`]: https://doc.rust-lang.org/stable/std/usize/constant.MAX.html
    /// [`Error::DimensionsTooLarge`]: enum.Error.html#variant.DimensionsTooLarge
    /// [`get_length_from_rows_and_columns_unchecked`]: #method.get_length_from_rows_and_columns_unchecked
    fn get_length_from_rows_and_columns(
        rows: NonZeroUsize,
        columns: NonZeroUsize,
    ) -> Result<usize> {
        match rows.get().checked_mul(columns.get()) {
            Some(length) => Ok(length),
            None => Err(Error::DimensionsTooLarge),
        }
    }

    /// Get the length of the data vector based on the number of rows and columns.
    ///
    /// This method will not check if the resulting length would exceed the maximum size of the
    /// matrix, [`::std::usize::MAX`]. If this happens, the method will panic due to an attempt to
    /// multiply with an overflow.
    ///
    /// If you cannot guarantee that the result will not exceed the maximum size, use
    /// [`get_length_from_rows_and_columns`] instead.
    ///
    /// # Safety
    ///
    /// If the product of `rows` and `length` would be greater than [`::std::usize::MAX`], the
    /// method will panic in debug builds and will silently overflow in release builds.
    ///
    /// [`::std::usize::MAX`]: https://doc.rust-lang.org/stable/std/usize/constant.MAX.html
    /// [`get_length_from_rows_and_columns`]: #method.get_length_from_rows_and_columns
    unsafe fn get_length_from_rows_and_columns_unchecked(
        rows: NonZeroUsize,
        columns: NonZeroUsize,
    ) -> usize {
        rows.get() * columns.get()
    }

    /// Get the number of rows in the matrix.
    pub fn get_rows(&self) -> usize {
        self.rows.get()
    }

    // endregion

    // region Element Operations

    /// Mutate each element in the matrix in place as given by the closure `mapping`.
    ///
    /// The `mapping` closure has three parameters, in this order:
    ///
    /// 1. The value of the current element as a mutable reference.
    /// 2. The row of the current element.
    /// 3. The column of the current element.
    ///
    /// # Example
    ///
    /// Convert a matrix of temperatures in °C to K:
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use reural_network::matrix::Matrix;
    ///
    /// let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
    /// let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
    /// let temperatures: [f64; 6] = [0.0, 10.0, 25.0, 50.0, 75.0, 100.0];
    /// let mut matrix: Matrix<f64> = Matrix::from_slice(rows, columns, &temperatures).unwrap();
    ///
    /// // Convert Celsius to Kelvin.
    /// matrix.map_ref_mut(|celsius, _row, _column| *celsius += 273.15);
    /// assert_eq!(matrix.as_slice(), [273.15, 283.15, 298.15, 323.15, 348.15, 373.15]);
    /// ```
    pub fn map_ref_mut<F>(&mut self, mut mapping: F)
    where
        F: FnMut(&mut T, usize, usize),
    {
        for row in 0..self.get_rows() {
            for column in 0..self.get_columns() {
                unsafe {
                    // Since we iterate over all rows and columns, they are always valid and we
                    // don't have to check any invariants.
                    let index: usize = self.get_index_unchecked(row, column);
                    mapping(&mut self.data[index], row, column);
                }
            }
        }
    }

    // endregion
}

impl<T> Matrix<T>
where
    T: Copy,
{
    // region Initialization

    /// Create a new matrix with the given dimensions and the given default value in all elements.
    ///
    /// The product of the number of `rows` and the number of `columns` must not exceed the maximum
    /// `usize` value, [`::std::usize::MAX`]. Otherwise, an [`Error::DimensionsTooLarge`] will be
    /// returned.
    ///
    /// # Example
    ///
    /// A `2x3` matrix with a default value of `0.25` for all elements can be created with the
    /// following lines of code:
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use reural_network::matrix::Matrix;
    ///
    /// let rows = NonZeroUsize::new(2).unwrap();
    /// let columns = NonZeroUsize::new(3).unwrap();
    /// let matrix: Matrix<f64> = Matrix::new(rows, columns, 0.25).unwrap();
    /// ```
    ///
    /// [`::std::usize::MAX`]: https://doc.rust-lang.org/stable/std/usize/constant.MAX.html
    /// [`Error::DimensionsTooLarge`]: enum.Error.html#variant.DimensionsTooLarge
    pub fn new(rows: NonZeroUsize, columns: NonZeroUsize, default: T) -> Result<Matrix<T>> {
        // Create the data structure and initialize it with the default value.
        let length: usize = Matrix::<T>::get_length_from_rows_and_columns(rows, columns)?;
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
    /// The product of the number of `rows` and the number of `columns` must not exceed the maximum
    /// `usize` value, [`::std::usize::MAX`]. If it does, an [`Error::DimensionsTooLarge`] will be
    /// returned. Furthermore, the product must be equal to the length of the given data slice.
    /// Otherwise, an [`Error::DimensionMismatch`] will be returned.
    ///
    /// # Example
    ///
    /// A `2x3` matrix can be created from a slice of length `6` with the following lines of code:
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use reural_network::matrix::Matrix;
    ///
    /// let rows = NonZeroUsize::new(2).unwrap();
    /// let columns = NonZeroUsize::new(3).unwrap();
    /// let matrix: Matrix<i32> = Matrix::from_slice(rows, columns, &[0, 1, 2, 3, 4, 5]).unwrap();
    /// ```
    ///
    /// This will produce the matrix:
    ///
    /// ```text
    /// [0 1 2]
    /// [3 4 5]
    /// ```
    ///
    /// [`::std::usize::MAX`]: https://doc.rust-lang.org/stable/std/usize/constant.MAX.html
    /// [`Error::DimensionMismatch`]: enum.Error.html#variant.DimensionMismatch
    /// [`Error::DimensionsTooLarge`]: enum.Error.html#variant.DimensionsTooLarge
    pub fn from_slice(rows: NonZeroUsize, columns: NonZeroUsize, data: &[T]) -> Result<Matrix<T>> {
        // Check that the length of the data slice matches the dimensions of the matrix.
        let length: usize = Matrix::<T>::get_length_from_rows_and_columns(rows, columns)?;
        if length != data.len() {
            return Err(Error::DimensionMismatch);
        }

        // Return the matrix.
        Ok(Matrix {
            rows,
            columns,
            data: data.to_vec(),
        })
    }

    // endregion

    // region Getters

    /// Get the value in the given `row` and `column`.
    ///
    /// If the `row` or `column` value is larger than the number of rows or columns in the matrix,
    /// respectively, an [`Error::CellOutOfBounds`] will be returned.
    ///
    /// If it can be guaranteed that the `row` and `column` values do not exceed the dimensions of
    /// the matrix, you can also use [`get_unchecked`].
    ///
    /// [`get_unchecked`]: #method.get_unchecked
    /// [`Error::CellOutOfBounds`]: enum.Error.html#variant.CellOutOfBounds
    pub fn get(&self, row: usize, column: usize) -> Result<T> {
        if row >= self.get_rows() || column >= self.get_columns() {
            return Err(Error::CellOutOfBounds);
        }

        unsafe { Ok(self.get_unchecked(row, column)) }
    }

    /// Get the value in the given `row` and `column`.
    ///
    /// This method does not check if the row and column parameters have valid values. If they are
    /// greater than or equal to the dimensions of the matrix, the resulting index will be out of
    /// bounds and the call will panic.
    ///
    /// This method should only be used if it is guaranteed by the caller that the row and column
    /// are within the dimensions of the matrix. If this cannot be guaranteed, use [`get`] instead.
    ///
    /// # Safety
    ///
    /// If the row or column are out of bounds of this matrix, an invalid index in the internal
    /// data structure will be accessed. This will cause the method to panic.
    ///
    /// [`get`]: #method.get
    pub unsafe fn get_unchecked(&self, row: usize, column: usize) -> T {
        self.data[self.get_index_unchecked(row, column)]
    }

    // endregion

    // region Element Operations

    /// Map each element in the matrix to a new element as given by the closure `mapping`.
    ///
    /// The `mapping` closure has three parameters, in this order:
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
    /// use reural_network::matrix::Matrix;
    ///
    /// let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
    /// let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
    /// let temperatures: [usize; 6] = [0, 10, 25, 50, 75, 100];
    /// let mut matrix: Matrix<usize> = Matrix::from_slice(rows, columns, &temperatures).unwrap();
    ///
    /// // Convert Celsius to Fahrenheit (these values come out as perfect integers).
    /// matrix.map(|celsius, _row, _column| (celsius * 9 / 5) + 32);
    /// assert_eq!(matrix.as_slice(), [32, 50, 77, 122, 167, 212]);
    /// ```
    pub fn map<F>(&mut self, mapping: F)
    where
        F: Fn(T, usize, usize) -> T,
    {
        for row in 0..self.get_rows() {
            for column in 0..self.get_columns() {
                unsafe {
                    // Since we iterate over all rows and columns, they are always valid and we
                    // don't have to check any invariants.
                    let index: usize = self.get_index_unchecked(row, column);
                    self.data[index] = mapping(self.data[index], row, column);
                }
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
    /// use reural_network::matrix::Matrix;
    ///
    /// let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
    /// let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
    /// let matrix: Matrix<usize> = Matrix::from_slice(rows, columns, &[0, 1, 2, 3, 4, 5]).unwrap();
    ///
    /// let transposed: Matrix<usize> = matrix.transpose();
    /// assert_eq!(transposed.get_rows(), 3);
    /// assert_eq!(transposed.get_columns(), 2);
    /// assert_eq!(transposed.as_slice(), &[0, 3, 1, 4, 2, 5]);
    /// ```
    pub fn transpose(&self) -> Matrix<T> {
        // The rows and columns are switched in the transposed matrix.
        let rows: NonZeroUsize = self.columns;
        let columns: NonZeroUsize = self.rows;

        // Allocate the required memory at once. This is faster than having to resize the vector
        // every few insertions.
        unsafe {
            // The rows and columns did not exceed the maximum size in the original matrix, so they
            // won't do this here, either.
            let length: usize =
                Matrix::<T>::get_length_from_rows_and_columns_unchecked(rows, columns);
            let mut data: Vec<T> = Vec::with_capacity(length);
            for index in 0..length {
                // Basically, iterate over the new data vector (which is still empty in the
                // beginning). For every index of the new vector, find the corresponding value from
                // the original matrix based on the index.

                // Get the row and column for this index in the transposed matrix.
                let row: usize = index / columns.get();
                let column: usize = index % columns.get();

                // Rows and columns are switched in the transposed matrix, so consider this when
                // getting the index for the original data.
                // Since we iterate over the vector and compute the row and column from this index,
                // the values are always valid.
                let value: T = self.get_unchecked(column, row);
                data.push(value)
            }

            Matrix {
                rows,
                columns,
                data,
            }
        }
    }

    // endregion
}

impl<T> Matrix<T>
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Copy,
{
    /// Compute the matrix product of `self` and `other` and return the result.
    ///
    /// The number of columns in `self` must be equal to the number of rows in `other`. Otherwise,
    /// [`Error::DimensionMismatch`] will be returned.
    ///
    /// The resulting matrix will have the dimensions `self.rows x other.columns`. If these
    /// dimensions would exceed the maximum size of matrices, [`Error::DimensionsTooLarge`] will be
    /// returned.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::num::NonZeroUsize;
    /// # use reural_network::Result;
    /// # use reural_network::matrix::Matrix;
    /// #
    /// // `m1` (2x3):
    /// // [1   2   3]
    /// // [4   5   6]
    /// let rows_m1 = NonZeroUsize::new(2).unwrap();
    /// let columns_m1 = NonZeroUsize::new(3).unwrap();
    /// let data_m1: [usize; 6] = [1, 2, 3, 4, 5, 6];
    /// let m1: Matrix<usize> = Matrix::from_slice(rows_m1, columns_m1, &data_m1).unwrap();
    ///
    /// // `m2` (3x2):
    /// // [7    8 ]
    /// // [9    10]
    /// // [11   12]
    /// let rows_m2 = NonZeroUsize::new(3).unwrap();
    /// let columns_m2 = NonZeroUsize::new(2).unwrap();
    /// let data_m2: [usize; 6] = [7, 8, 9, 10, 11, 12];
    /// let m2: Matrix<usize> = Matrix::from_slice(rows_m2, columns_m2, &data_m2).unwrap();
    ///
    /// // Calculate `m3` as the matrix product of `m1` and `m2`.
    /// // `m3` (2x2):
    /// // [58    64 ]
    /// // [139   154]
    /// let m3: Matrix<usize> = m1.matrix_mul(&m2).unwrap();
    /// assert_eq!(m3.get_rows(), 2);
    /// assert_eq!(m3.get_columns(), 2);
    /// assert_eq!(m3.as_slice(), &[58, 64, 139, 154]);
    /// ```
    ///
    /// [`Error::DimensionMismatch`]: enum.Error.html#variant.DimensionMismatch
    /// [`Error::DimensionsTooLarge`]: enum.Error.html#variant.DimensionsTooLarge
    pub fn matrix_mul(&self, other: &Matrix<T>) -> Result<Matrix<T>> {
        if self.get_columns() != other.get_rows() {
            return Err(Error::DimensionMismatch);
        }

        // Ensure that the dimensions of the result matrix do not exceed the maximum size.
        let rows: NonZeroUsize = self.rows;
        let columns: NonZeroUsize = other.columns;
        let size: usize = Matrix::<T>::get_length_from_rows_and_columns(rows, columns)?;

        let mut result = Matrix {
            rows,
            columns,
            data: Vec::with_capacity(size),
        };

        for row in 0..result.get_rows() {
            for column in 0..result.get_columns() {
                // All row and column values are valid so it is safe to use these unsafe and
                // unchecked methods.
                unsafe {
                    // Calculate the sum of products. Since there is no general neutral element
                    // of addition for `T` (e.g., 0 would be one for all number types), calculate
                    // the first product outside the loop to initialize the variable without special
                    // cases inside the loop. There must be at least this first element since we can
                    // not have matrices without any elements.
                    let mut element: T =
                        self.get_unchecked(row, 0) * other.get_unchecked(0, column);

                    for i in 1..self.get_columns() {
                        let product: T =
                            self.get_unchecked(row, i) * other.get_unchecked(i, column);

                        // We don't want to require `T` to implement `AddAssign`, but only the
                        // simpler `Add`.
                        element = element + product;
                    }

                    // Set the element in the result matrix. Since we are iterating over the
                    // elements in row-major format, the index at which `element` will be inserted
                    // will be correct.
                    result.data.push(element);
                }
            }
        }

        Ok(result)
    }
}

impl Matrix<f64> {
    // region Initialization

    /// Create a new matrix with the given dimensions and random elements in the inclusive range
    /// `[0.0, 1.0]` (i.e., including both `0.0` and `1.0`).
    ///
    /// The product of the number of `rows` and the number of `columns` must not exceed the maximum
    /// `usize` value, [`::std::usize::MAX`]. Otherwise, an [`Error::DimensionsTooLarge`] will be
    /// returned.
    ///
    /// # Example
    ///
    /// A `2x3` matrix with a default value of `0.25` for all elements can be created with the
    /// following lines of code:
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use reural_network::matrix::Matrix;
    ///
    /// let rows = NonZeroUsize::new(2).unwrap();
    /// let columns = NonZeroUsize::new(3).unwrap();
    /// let matrix: Matrix<f64> = Matrix::from_random(rows, columns).unwrap();
    /// ```
    ///
    /// [`::std::usize::MAX`]: https://doc.rust-lang.org/stable/std/usize/constant.MAX.html
    /// [`Error::DimensionsTooLarge`]: enum.Error.html#variant.DimensionsTooLarge
    pub fn from_random(rows: NonZeroUsize, columns: NonZeroUsize) -> Result<Matrix<f64>> {
        // Get random data in the range of [0.0, 1.0].
        let length: usize = Matrix::<f64>::get_length_from_rows_and_columns(rows, columns)?;
        let mut rng: ThreadRng = thread_rng();
        let mut data: Vec<f64> = Vec::with_capacity(length);
        data.resize_with(length, || rng.sample(Uniform::new_inclusive(0.0, 1.0)));

        // Return the matrix.
        Ok(Matrix {
            rows,
            columns,
            data,
        })
    }

    // endregion
}

impl<T> Display for Matrix<T>
where
    T: Display,
{
    /// Get a human readable representation of this matrix.
    ///
    /// The matrix will be formatted in a rectangular array with the dimensions of the matrix.
    ///
    /// # Example
    ///
    /// A `2x3` matrix with some data as produced by the code
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use reural_network::matrix::Matrix;
    ///
    /// let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
    /// let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
    /// let data: [f64; 6] = [0.25, 1.33, -0.1, 1.0, -2.73, 1.2];
    /// let matrix: Matrix<f64> = Matrix::from_slice(rows, columns, &data).unwrap();
    /// ```
    ///
    /// will be formatted to the following text (e.g. when using [`println!`] to print to the
    /// console):
    ///
    /// ```text
    /// [0.25   1.33    -0.1]
    /// [1      -2.73   1.2 ]
    /// ```
    ///
    /// [`println!`]: https://doc.rust-lang.org/stable/std/macro.println.html
    fn fmt(&self, formatter: &mut Formatter) -> ::std::fmt::Result {
        // Align all columns, but each column may have a different alignment. Thus, first iterate
        // over the columns, then the rows, to get the width of each column from all values in the
        // column.
        let mut column_widths: Vec<usize> = Vec::with_capacity(self.get_columns());
        for column in 0..self.get_columns() {
            // Get the maximum width of the current column.
            let mut max_width: usize = 0;
            for row in 0..self.get_rows() {
                // Do not use self.get_unchecked() here as this requires T to implement Copy.
                unsafe {
                    // We iterate over the rows and columns and thus, they are always valid.
                    let value: String =
                        format!("{}", self.data[self.get_index_unchecked(row, column)]);
                    max_width = max(max_width, value.len());
                }
            }

            // Remember the current column's width.
            column_widths.push(max_width);
        }

        // Now, go through each row and format each value with the width of its column.
        let mut rows: Vec<String> = Vec::with_capacity(self.get_rows());
        for row in 0..self.get_rows() {
            // For each row, collect the formatted values first.
            let mut row_values: Vec<String> = Vec::with_capacity(self.get_columns());
            for (column, width) in column_widths.iter().enumerate() {
                unsafe {
                    // We iterate over the rows and columns and thus, they are always valid.
                    let value: String = format!(
                        "{:<width$}", // Left-align all values.
                        // Do not use self.get_unchecked() here as this requires T to implement
                        // Copy.
                        self.data[self.get_index_unchecked(row, column)],
                        width = width
                    );

                    row_values.push(value);
                }
            }

            // Concatenate all aligned values in the row with three spaces. Surround the values with
            // square brackets.
            rows.push(format!("[{}]", row_values.join("   ")));
        }

        // Concatenate all rows with a new line.
        write!(formatter, "{}", rows.join("\n"))
    }
}

impl_scalar_assign_operators!();
impl_element_wise_binary_operators!();
impl_scalar_binary_operators!();
impl_unary_operators!();

#[cfg(test)]
mod tests {

    use super::*;
    use crate::test_element_wise_binary_operators;
    use crate::test_scalar_assign_operators;
    use crate::test_scalar_binary_operators;
    use crate::test_unary_operators;

    // region Initialization

    /// Test creating a new matrix with dimensions that are not exceeding the maximum size.
    #[test]
    fn new_valid_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(5).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let matrix_result: Result<Matrix<usize>> = Matrix::new(rows, columns, 0);

        assert!(matrix_result.is_ok());

        let matrix: Matrix<usize> = matrix_result.unwrap();
        assert_eq!(matrix.rows.get(), rows.get());
        assert_eq!(matrix.columns.get(), columns.get());
        assert_eq!(matrix.as_slice(), [0_usize; 15]);
    }

    /// Test creating a new matrix with dimensions that exceed the maximum size.
    #[test]
    fn new_exceeding_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(::std::usize::MAX).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let matrix_result: Result<Matrix<usize>> = Matrix::new(rows, columns, 0);

        assert!(matrix_result.is_err());

        let is_correct_error: bool = match matrix_result.unwrap_err() {
            Error::DimensionsTooLarge => true,
            _ => false,
        };

        assert!(
            is_correct_error,
            "Expected error Error::DimensionsTooLarge not satisfied."
        );
    }

    /// Test creating a new matrix with random data with dimensions that do not exceed the maximum
    /// size.
    #[test]
    fn from_random_valid_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(5).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let matrix_result: Result<Matrix<f64>> = Matrix::from_random(rows, columns);

        assert!(matrix_result.is_ok());

        let matrix: Matrix<f64> = matrix_result.unwrap();
        assert_eq!(matrix.rows.get(), rows.get());
        assert_eq!(matrix.columns.get(), columns.get());

        let data: &[f64] = matrix.as_slice();
        assert_eq!(data.len(), 15);
        for element in data.iter() {
            assert!(*element >= 0.0);
            assert!(*element <= 1.0);
        }
    }

    /// Test creating a new matrix with random data with dimensions that not exceed the maximum
    /// size.
    #[test]
    fn from_random_invalid_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(::std::usize::MAX).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let matrix_result: Result<Matrix<f64>> = Matrix::from_random(rows, columns);

        assert!(matrix_result.is_err());

        let is_correct_error: bool = match matrix_result.unwrap_err() {
            Error::DimensionsTooLarge => true,
            _ => false,
        };

        assert!(
            is_correct_error,
            "Expected error Error::DimensionsTooLarge not satisfied."
        );
    }

    /// Test creating a new matrix from a slice with dimensions that do not exceed the maximum size
    /// and that match the length of the given slice.
    #[test]
    fn from_slice_valid_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(5).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [usize; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let matrix_result: Result<Matrix<usize>> = Matrix::from_slice(rows, columns, &data);

        assert!(matrix_result.is_ok());

        let matrix: Matrix<usize> = matrix_result.unwrap();
        assert_eq!(matrix.rows.get(), rows.get());
        assert_eq!(matrix.columns.get(), columns.get());
        assert_eq!(matrix.as_slice(), data);
    }

    /// Test creating a new matrix from a slice with dimensions that exceed the maximum size.
    #[test]
    fn from_slice_exceeding_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(::std::usize::MAX).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let data: [usize; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let matrix_result: Result<Matrix<usize>> = Matrix::from_slice(rows, columns, &data);

        assert!(matrix_result.is_err());

        let is_correct_error: bool = match matrix_result.unwrap_err() {
            Error::DimensionsTooLarge => true,
            _ => false,
        };

        assert!(
            is_correct_error,
            "Expected error Error::DimensionsTooLarge not satisfied."
        );
    }

    /// Test creating a new matrix from a slice with dimensions that do not match the length of the
    /// given slice.
    #[test]
    fn from_slice_not_matching_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(5).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [usize; 5] = [0, 1, 2, 3, 4];
        let matrix_result: Result<Matrix<usize>> = Matrix::from_slice(rows, columns, &data);

        assert!(matrix_result.is_err());

        let is_correct_error: bool = match matrix_result.unwrap_err() {
            Error::DimensionMismatch => true,
            _ => false,
        };

        assert!(
            is_correct_error,
            "Expected error Error::DimensionMismatch not satisfied."
        );
    }

    // endregion

    // region Getters

    /// Test getting a slice of the matrix data.
    #[test]
    fn as_slice() {
        let data: [usize; 6] = [0, 10, 20, 30, 40, 50];
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let matrix: Matrix<usize> = Matrix::from_slice(rows, columns, &data).unwrap();

        assert_eq!(matrix.as_slice(), &data);
    }

    /// Test getting the number of columns.
    #[test]
    fn get_columns() {
        let rows: usize = 3;
        let columns: usize = 2;
        let matrix = Matrix {
            rows: NonZeroUsize::new(rows).unwrap(),
            columns: NonZeroUsize::new(columns).unwrap(),
            data: vec![0, 1],
        };

        assert_eq!(matrix.get_columns(), columns);
    }

    /// Test getting the unchecked index for given rows and columns.
    #[test]
    fn get_index_unchecked() {
        let rows: NonZeroUsize = NonZeroUsize::new(10).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(10).unwrap();
        let matrix: Matrix<usize> = Matrix::new(rows, columns, 0).unwrap();

        unsafe {
            // (0, 0) => 0
            assert_eq!(matrix.get_index_unchecked(0, 0), 0);

            // (0, 1) => 1
            assert_eq!(matrix.get_index_unchecked(0, 1), 1);

            // (1, 0) => 10
            assert_eq!(matrix.get_index_unchecked(1, 0), 10);

            // (3, 7) => 37
            assert_eq!(matrix.get_index_unchecked(3, 7), 37);

            // (9, 9) => 99
            assert_eq!(matrix.get_index_unchecked(9, 9), 99);

            // (10, 0) => 100 (out of bounds)
            assert_eq!(matrix.get_index_unchecked(10, 0), 100);
        }
    }

    /// Test that getting get index when iterating over all elements in row-major format will yield
    /// indices that are increasing exactly by 1.
    #[test]
    fn get_index_unchecked_correct_increments() {
        let rows = NonZeroUsize::new(5).unwrap();
        let columns = NonZeroUsize::new(7).unwrap();
        let matrix: Matrix<usize> = Matrix::new(rows, columns, 0).unwrap();

        let mut previous_index: usize = 0;
        for row in 0..matrix.get_rows() {
            for column in 0..matrix.get_columns() {
                unsafe {
                    let index: usize = matrix.get_index_unchecked(row, column);

                    // For the very first element e_0,0, the expected index is, of course, 0. For
                    // all other elements, the expected index is the previous index plus 1.
                    let expected_index: usize = match (row, column) {
                        (0, 0) => 0,
                        _ => previous_index + 1,
                    };

                    assert_eq!(index, expected_index, "row {}, column {}", row, column);

                    previous_index = index;
                }
            }
        }
    }

    /// Test getting the length of the data vector based on the number of rows and columns in the
    /// matrix when the product of the number of rows and columns does not overflow.
    #[test]
    fn get_length_from_rows_and_columns_non_overflowing() {
        let rows: NonZeroUsize = NonZeroUsize::new(7).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(6).unwrap();
        let length: Result<usize> =
            Matrix::<usize>::get_length_from_rows_and_columns(rows, columns);

        assert!(length.is_ok());
        assert_eq!(length.unwrap(), rows.get() * columns.get());
    }

    /// Test getting the length of the data vector based on the number of rows and columns in the
    /// matrix when the product of the number of rows and columns would overflow.
    #[test]
    fn get_length_from_rows_and_columns_overflowing() {
        let rows: NonZeroUsize = NonZeroUsize::new(::std::usize::MAX).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let length: Result<usize> =
            Matrix::<usize>::get_length_from_rows_and_columns(rows, columns);

        assert!(length.is_err());

        let is_correct_error: bool = match length.unwrap_err() {
            Error::DimensionsTooLarge => true,
            _ => false,
        };

        assert!(
            is_correct_error,
            "Expected error Error::DimensionsTooLarge not satisfied."
        );
    }

    /// Test getting the length of the data vector based on the number of rows and columns in the
    /// matrix when the product of the number of rows and columns does not overflow, without
    /// checking if the length would exceed the maximum size.
    #[test]
    fn get_length_from_rows_and_columns_unchecked_non_overflowing() {
        let rows: NonZeroUsize = NonZeroUsize::new(7).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(6).unwrap();
        unsafe {
            let length: usize =
                Matrix::<usize>::get_length_from_rows_and_columns_unchecked(rows, columns);

            assert_eq!(length, rows.get() * columns.get());
        }
    }

    /// Test getting the length of the data vector based on the number of rows and columns in the
    /// matrix when the product of the number of rows and columns would overflow, without checking
    /// if the length would exceed the maximum size.
    ///
    /// In debug mode, the overflow will cause a panic.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "attempt to multiply with overflow")]
    fn get_length_from_rows_and_columns_unchecked_overflowing_debug() {
        let rows: NonZeroUsize = NonZeroUsize::new(::std::usize::MAX - 1).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        unsafe {
            let _ = Matrix::<usize>::get_length_from_rows_and_columns_unchecked(rows, columns);
        }
    }

    /// Test getting the length of the data vector based on the number of rows and columns in the
    /// matrix when the product of the number of rows and columns would overflow, without checking
    /// if the length would exceed the maximum size.
    ///
    /// In release mode, the computation will silently overflow.
    #[test]
    #[cfg(not(debug_assertions))]
    fn get_length_from_rows_and_columns_unchecked_overflowing_release() {
        let rows: NonZeroUsize = NonZeroUsize::new(::std::usize::MAX - 1).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        unsafe {
            let length: usize =
                Matrix::<usize>::get_length_from_rows_and_columns_unchecked(rows, columns);

            assert_eq!(length, ::std::usize::MAX - 3);
        }
    }

    /// Test getting the number of rows.
    #[test]
    fn get_rows() {
        let rows: usize = 3;
        let columns: usize = 2;
        let matrix = Matrix {
            rows: NonZeroUsize::new(rows).unwrap(),
            columns: NonZeroUsize::new(columns).unwrap(),
            data: vec![0, 1],
        };

        assert_eq!(matrix.get_rows(), rows);
    }

    /// Test getting a value when the row and column are valid.
    #[test]
    fn get_valid_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [u64; 6] = [10, 11, 12, 13, 14, 15];
        let matrix: Matrix<u64> = Matrix::from_slice(rows, columns, &data).unwrap();

        let value: Result<u64> = matrix.get(0, 0);
        assert!(value.is_ok());
        assert_eq!(value.unwrap(), data[0]);
    }

    /// Test getting a value when the row or column are invalid.
    #[test]
    fn get_invalid_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [u64; 6] = [10, 11, 12, 13, 14, 15];
        let matrix: Matrix<u64> = Matrix::from_slice(rows, columns, &data).unwrap();

        // Both the row and column are invalid.
        let value: Result<u64> = matrix.get(rows.get() + 1, columns.get() + 2);
        assert!(value.is_err());

        let is_correct_error: bool = match value.unwrap_err() {
            Error::CellOutOfBounds => true,
            _ => false,
        };

        assert!(
            is_correct_error,
            "Expected error Error::CellOutOfBounds not satisfied."
        );

        // Only the row is invalid.
        let value: Result<u64> = matrix.get(rows.get() + 1, columns.get());
        assert!(value.is_err());

        let is_correct_error: bool = match value.unwrap_err() {
            Error::CellOutOfBounds => true,
            _ => false,
        };

        assert!(
            is_correct_error,
            "Expected error Error::CellOutOfBounds not satisfied."
        );

        // Only the column is invalid.
        let value: Result<u64> = matrix.get(rows.get(), columns.get() + 2);
        assert!(value.is_err());

        let is_correct_error: bool = match value.unwrap_err() {
            Error::CellOutOfBounds => true,
            _ => false,
        };

        assert!(
            is_correct_error,
            "Expected error Error::CellOutOfBounds not satisfied."
        );
    }

    /// Test getting a value without checking the row and column when the row and column are valid.
    #[test]
    fn get_unchecked_valid_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [u64; 6] = [10, 11, 12, 13, 14, 15];
        let matrix: Matrix<u64> = Matrix::from_slice(rows, columns, &data).unwrap();

        unsafe {
            assert_eq!(matrix.get_unchecked(0, 0), 10);
            assert_eq!(matrix.get_unchecked(0, 1), 11);
            assert_eq!(matrix.get_unchecked(0, 2), 12);
            assert_eq!(matrix.get_unchecked(1, 0), 13);
            assert_eq!(matrix.get_unchecked(1, 1), 14);
            assert_eq!(matrix.get_unchecked(1, 2), 15);
        }
    }

    /// Test getting a value without checking the row and column when the row or column are invalid.
    #[test]
    #[should_panic(expected = "index out of bounds: the len is 6 but the index is 11")]
    fn get_unchecked_invalid_dimensions() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [u64; 6] = [10, 11, 12, 13, 14, 15];
        let matrix: Matrix<u64> = Matrix::from_slice(rows, columns, &data).unwrap();

        unsafe {
            let _: u64 = matrix.get_unchecked(rows.get(), columns.get() + 2);
        }
    }

    // endregion

    // region Element Operations

    /// Test mapping the data in a matrix.
    #[test]
    fn map() {
        // Temperature in °C.
        let temperatures: [usize; 6] = [0, 10, 25, 50, 75, 100];

        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let mut temperature: Matrix<usize> =
            Matrix::from_slice(rows, columns, &temperatures).unwrap();

        // Convert Celsius to Fahrenheit (the values come out as perfect integers).
        temperature.map(|celsius, _row, _column| (celsius * 9 / 5) + 32);

        // Temperature in °F.
        assert_eq!(temperature.as_slice(), [32, 50, 77, 122, 167, 212]);
    }

    /// Test mapping the data in a matrix by mutable reference.
    #[test]
    fn map_ref_mut() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let temperatures: [f64; 6] = [0.0, 10.0, 25.0, 50.0, 75.0, 100.0];
        let mut matrix: Matrix<f64> = Matrix::from_slice(rows, columns, &temperatures).unwrap();

        // Convert Celsius to Kelvin.
        matrix.map_ref_mut(|celsius, _row, _column| *celsius += 273.15);
        assert_eq!(
            matrix.as_slice(),
            [273.15, 283.15, 298.15, 323.15, 348.15, 373.15]
        );
    }

    /// Test transposing a matrix.
    #[test]
    fn transpose() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [usize; 6] = [0, 1, 2, 3, 4, 5];
        let matrix: Matrix<usize> = Matrix::from_slice(rows, columns, &data).unwrap();

        let transposed: Matrix<usize> = matrix.transpose();
        assert_eq!(transposed.get_rows(), columns.get());
        assert_eq!(transposed.get_columns(), rows.get());
        assert_eq!(transposed.as_slice(), [0, 3, 1, 4, 2, 5]);
    }

    /// Test matrix multiplication when the dimensions of the matrix are correct.
    #[test]
    fn matrix_mul_correct_dimensions() {
        let rows_m1 = NonZeroUsize::new(1).unwrap();
        let columns_m1 = NonZeroUsize::new(3).unwrap();
        let data_m1: [usize; 3] = [3, 4, 2];
        let m1: Matrix<usize> = Matrix::from_slice(rows_m1, columns_m1, &data_m1).unwrap();

        let rows_m2 = NonZeroUsize::new(3).unwrap();
        let columns_m2 = NonZeroUsize::new(4).unwrap();
        let data_m2: [usize; 12] = [13, 9, 7, 15, 8, 7, 4, 6, 6, 4, 0, 3];
        let m2: Matrix<usize> = Matrix::from_slice(rows_m2, columns_m2, &data_m2).unwrap();

        let result: Result<Matrix<usize>> = m1.matrix_mul(&m2);
        assert!(result.is_ok());

        let m3: Matrix<usize> = result.unwrap();
        assert_eq!(m3.get_rows(), 1);
        assert_eq!(m3.get_columns(), 4);
        assert_eq!(m3.as_slice(), &[83, 63, 37, 75]);
    }

    /// Test matrix multiplication when the dimensions of the matrix are incorrect.
    #[test]
    fn matrix_mul_incorrect_dimensions() {
        let rows_m1 = NonZeroUsize::new(1).unwrap();
        let columns_m1 = NonZeroUsize::new(3).unwrap();
        let data_m1: [usize; 3] = [3, 4, 2];
        let m1: Matrix<usize> = Matrix::from_slice(rows_m1, columns_m1, &data_m1).unwrap();

        let rows_m2 = NonZeroUsize::new(4).unwrap();
        let columns_m2 = NonZeroUsize::new(3).unwrap();
        let data_m2: [usize; 12] = [13, 9, 7, 15, 8, 7, 4, 6, 6, 4, 0, 3];
        let m2: Matrix<usize> = Matrix::from_slice(rows_m2, columns_m2, &data_m2).unwrap();

        let result: Result<Matrix<usize>> = m1.matrix_mul(&m2);
        assert!(result.is_err());

        let is_correct_error: bool = match result.unwrap_err() {
            Error::DimensionMismatch => true,
            _ => false,
        };

        assert!(
            is_correct_error,
            "Expected error Error::DimensionMismatch not satisfied."
        );
    }

    // Test the operators.
    test_scalar_assign_operators!();
    test_element_wise_binary_operators!();
    test_scalar_binary_operators!();
    test_unary_operators!();

    // endregion

    // region Display

    /// Test formatting the matrix in debug mode.
    #[test]
    fn debug() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [f64; 6] = [0.25, 1.33, -0.1, 1.0, -2.73, 1.2];
        let matrix: Matrix<f64> = Matrix::from_slice(rows, columns, &data).unwrap();

        let debug: String = format!("{:?}", matrix);
        assert_eq!(
            "Matrix { rows: 2, columns: 3, data: [0.25, 1.33, -0.1, 1.0, -2.73, 1.2] }",
            debug
        );
    }

    /// Test formatting the matrix in a human readable way.
    #[test]
    fn display() {
        let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
        let data: [f64; 6] = [0.25, 1.33, -0.1, 1.0, -2.73, 1.2];
        let matrix: Matrix<f64> = Matrix::from_slice(rows, columns, &data).unwrap();

        // This should come out as:
        // [0.25   1.33    -0.1]
        // [1      -2.73   1.2 ]
        let display: String = format!("{}", matrix);
        assert_eq!("[0.25   1.33    -0.1]\n[1      -2.73   1.2 ]", display);
    }

    // endregion
}
