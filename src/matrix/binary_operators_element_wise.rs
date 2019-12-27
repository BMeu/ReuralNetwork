// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Macros to implement element-wise binary operations.
//!
//! The main macros in this module are [`impl_element_wise_binary_operators`] to implement all
//! binary operators as element-wise operations, and [`test_element_wise_binary_operators`] to test
//! these implementations.
//!
//! [`impl_element_wise_binary_operators`]: ../../macro.impl_element_wise_binary_operators.html
//! [`test_element_wise_binary_operators`]: ../../macro.test_element_wise_binary_operators.html

// region Implement

/// Implement all binary operators as element-wise operations on two matrices `Matrix<T>` and all
/// possible combinations including (immutable) references of these types.
///
/// # Implemented Binary Operators Traits
///
/// * [`Add`]
/// * [`BitAnd`]
/// * [`BitOr`]
/// * [`BitXor`]
/// * [`Div`]
/// * [`Mul`]
/// * [`Rem`]
/// * [`Shl`]
/// * [`Shr`]
/// * [`Sub`]
///
/// All these traits must be `use`d in the module calling the macro.
///
/// [`Add`]: https://doc.rust-lang.org/std/ops/trait.Add.html
/// [`BitAnd`]: https://doc.rust-lang.org/std/ops/trait.BitAnd.html
/// [`BitOr`]: https://doc.rust-lang.org/std/ops/trait.BitOr.html
/// [`BitXor`]: https://doc.rust-lang.org/std/ops/trait.BitXor.html
/// [`Div`]: https://doc.rust-lang.org/std/ops/trait.Div.html
/// [`Mul`]: https://doc.rust-lang.org/std/ops/trait.Mul.html
/// [`Rem`]: https://doc.rust-lang.org/std/ops/trait.Rem.html
/// [`Shl`]: https://doc.rust-lang.org/std/ops/trait.Shl.html
/// [`Shr`]: https://doc.rust-lang.org/std/ops/trait.Shr.html
/// [`Sub`]: https://doc.rust-lang.org/std/ops/trait.Sub.html
#[doc(hidden)]
#[macro_export]
macro_rules! impl_element_wise_binary_operators {
    () => {
        // Addition.
        $crate::impl_element_wise_binary_operator_with_references!(
            Add,
            add,
            +,
            "Add each element in `self` to the corresponding element in `other`.",
            f64,
            [0.25, 1.33, -0.1, 1.0, -2.73, 1.2],
            [0.25, 1.33, -0.1, 1.0, -2.73, 1.2],
            [0.5, 2.66, -0.2, 2.0, -5.46, 2.4]
        );

        // Bitwise AND.
        $crate::impl_element_wise_binary_operator_with_references!(
            BitAnd,
            bitand,
            &,
            "Calculate the bitwise AND of each element in `self` with the corresponding element in \
             `other`.",
            u8,
            [7, 0, 1, 3, 5, 9],
            [4, 15, 1, 6, 5, 7],
            [4, 0, 1, 2, 5, 1]
        );

        // Bitwise OR.
        $crate::impl_element_wise_binary_operator_with_references!(
            BitOr,
            bitor,
            |,
            "Calculate the bitwise OR of each element in `self` with the corresponding element in \
             `other`.",
            u8,
            [7, 0, 1, 3, 5, 9],
            [4, 15, 1, 6, 5, 7],
            [7, 15, 1, 7, 5, 15]
        );

        // Bitwise XOR.
        $crate::impl_element_wise_binary_operator_with_references!(
            BitXor,
            bitxor,
            ^,
            "Calculate the bitwise XOR of each element in `self` with the corresponding element in \
             `other`.",
            u8,
            [7, 0, 1, 3, 5, 9],
            [4, 15, 1, 6, 5, 7],
            [3, 15, 0, 5, 0, 14]
        );

        // Division.
        $crate::impl_element_wise_binary_operator_with_references!(
            Div,
            div,
            /,
            "Divide each element in `self` by the corresponding element in `other`.",
            f64,
            [1.0, 1.33, -0.1, 4.0, -2.73, 4.0],
            [2.0, 1.33, -4.0, -2.0, 2.73, 0.1],
            [0.5, 1.0, 0.025, -2.0, -1.0, 40.0]
        );

        // Multiplication.
        $crate::impl_element_wise_binary_operator_with_references!(
            Mul,
            mul,
            *,
            "Multiply each element in `self` to the corresponding element in `other`, i.e.\
             calculate the Hadamard product of `self` and `other`.",
            f64,
            [0.25, 1.0, -0.3, -1.0, 2.73, 1.2],
            [2.0, 0.25, -0.3, 2.0, -2.0, 1.2],
            [0.5, 0.25, 0.09, -2.0, -5.46, 1.44]
        );

        // Remainder.
        $crate::impl_element_wise_binary_operator_with_references!(
            Rem,
            rem,
            %,
            "Calculate the remainder when dividing each element in `self` by the corresponding \
             element in `other`.",
            i64,
            [2, 6, -3, 5, 5, -10],
            [1, 4, 2, -4, 6, -2],
            [0, 2, -1, 1, 5, 0]
        );

        // Bitwise left shift.
        $crate::impl_element_wise_binary_operator_with_references!(
            Shl,
            shl,
            <<,
            "Bitwise shift each element in `self` to the left by the corresponding element in \
             `other`.",
            u8,
            [7, 0, 1, 5, 6, 3],
            [1, 5, 5, 0, 2, 3],
            [14, 0, 32, 5, 24, 24]
        );

        // Bitwise right shift.
        $crate::impl_element_wise_binary_operator_with_references!(
            Shr,
            shr,
            >>,
            "Bitwise shift each element in `self` to the right by the corresponding element in \
             `other`.",
            u8,
            [7, 0, 1, 5, 6, 15],
            [1, 5, 5, 2, 2, 3],
            [3, 0, 0, 1, 1, 1]
        );

        // Subtraction.
        $crate::impl_element_wise_binary_operator_with_references!(
            Sub,
            sub,
            -,
            "Subtract each element in `other` from the corresponding element in `self`.",
            f64,
            [0.25, 1.0, -0.1, 1.0, -2.73, 1.3],
            [0.25, 0.4, 0.1, -1.0, -2.0, 3.6],
            [0.0, 0.6, -0.2, 2.0, -0.73, -2.3]
        );
    };
}

/// Implement a given binary operator as an element-wise operation on `Matrix<T>` with another
/// `Matrix<T>` and all possible combinations including (immutable) references of these types.
///
/// # Parameters
///
/// * `$trait`: The binary-operator trait to implement. This trait must also be implemented by `T`.
/// * `$fn`: The name of the function that implements the binary operator.
/// * `$operator`: The actual binary operator, e.g. `+` for the `Add` trait.
/// * `$explanation`: A short explanation for the documentation of what the operator does.
/// * `$data_type`: The type of the data for the first and second matrix in the documentation
///                 example.
/// * `$data_self`: The actual data array for the first (`self`) matrix in the documentation
///                 example. It must have a length of `6`.
/// * `$data_other`: The actual data array for the second (`other`) matrix in the documentation
///                  example. It must have a length of `6`.
/// * `$expected_result`: An array of expected values for the operation in the documentation
///                       example.
///
/// # Example
///
/// Implement addition::
///
/// ```text
/// impl_element_wise_binary_operator_with_references!(
///     Add,
///     add,
///     +,
///     "Add each element in `self` to the corresponding element in `other`.",
///     f64,
///     [0.25, 1.33, -0.1, 1.0, -2.73, 1.2],
///     [0.25, 1.33, -0.1, 1.0, -2.73, 1.2],
///     [0.5, 2.66, -0.2, 2.0, -5.46, 2.4]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! impl_element_wise_binary_operator_with_references {
    ($trait:tt,
     $fn:tt,
     $operator:tt,
     $explanation:expr,
     $data_type:ty,
     $data_self:expr,
     $data_other:expr,
     $result:expr
    ) => {
        // Implement the operator for Matrix<T> and Matrix<T>.
        $crate::impl_element_wise_binary_operator!(
            *,
            *,
            $trait,
            $fn,
            $operator,
            $crate::doc_element_wise_binary_operator!(
                $explanation,
                $data_type,
                $data_self,
                $data_other,
                matrix,
                $operator,
                other,
                $result
            )
        );

        // Implement the operator for Matrix<T> and &'_ Matrix<T>.
        $crate::impl_element_wise_binary_operator!(
            *,
            &,
            $trait,
            $fn,
            $operator,
            $crate::doc_element_wise_binary_operator!(
                $explanation,
                $data_type,
                $data_self,
                $data_other,
                matrix,
                $operator,
                &other,
                $result
            )
        );

        // Implement the operator for &'_ Matrix<T> and Matrix<T>.
        $crate::impl_element_wise_binary_operator!(
            &,
            *,
            $trait,
            $fn,
            $operator,
            $crate::doc_element_wise_binary_operator!(
                $explanation,
                $data_type,
                $data_self,
                $data_other,
                &matrix,
                $operator,
                other,
                $result
            )
        );

        // Implement the operator for &'_ Matrix<T> and &'_ Matrix<T>.
        $crate::impl_element_wise_binary_operator!(
            &,
            &,
            $trait,
            $fn,
            $operator,
            $crate::doc_element_wise_binary_operator!(
                $explanation,
                $data_type,
                $data_self,
                $data_other,
                &matrix,
                $operator,
                &other,
                $result
            )
        );
    };
}

/// Implement a given binary operator as an element-wise operation on a matrix whose element type
/// also implements the operator.
///
/// # Parameters
///
/// * `$lhs_access`: The left-hand side access type of the operator, either `*` for owned access or
///                  `&` for referenced access.
/// * `$rhs_access`: The right-hand side access type of the operator, either `*` for owned access or
///                  `&` for referenced access
/// * `$trait`: The binary-operator trait to implement. This trait must also be implemented by `T`.
/// * `$fn`: The name of the function that implements the binary operator.
/// * `$operator`: The actual binary operator, e.g. `+` for the `Add` trait.
/// * `$documentation`: The documentation for the operator method.
///
/// # Example
///
/// Implement addition for `Matrix<T>` to which a `&'_ Matrix<T>` is added:
///
/// ```text
/// impl_element_wise_binary_operator!(
///     *,
///     &,
///     Add,
///     add,
///     +,
///     "Element-wise add the values of `other` to `self`."
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! impl_element_wise_binary_operator {
    ($lhs_access:tt, $rhs_access:tt, $trait:tt, $fn:tt, $operator:tt, $documentation:expr) => {
        impl<T> $trait<$crate::specify_matrix_type!($rhs_access)>
        for $crate::specify_matrix_type!($lhs_access)
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = Result<Matrix<T>>;

            #[doc = $documentation]
            fn $fn(self, other: $crate::specify_matrix_type!($rhs_access)) -> Self::Output {
                // For element-wise operations, the dimensions of both matrices must be the same.
                if self.get_rows() != other.get_rows() || self.get_columns() != other.get_columns()
                {
                    return Err(Error::DimensionMismatch);
                }

                let mut result: Matrix<T> = Matrix {
                    rows: self.rows,
                    columns: self.columns,
                    data: self.data.clone(),
                };

                // The row and column are given by the map method and are thus valid.
                result.map(|element, row, column| unsafe {
                    element $operator other.get_unchecked(row, column)
                });

                Ok(result)
            }
        }
    };
}

// endregion

// region Tests

/// Implement tests for all element-wise binary operations on two matrices `Matrix<T>` and all
/// possible combinations including (immutable) references of these types.
///
/// # Tested Binary Operators Traits
///
/// * [`Add`]
/// * [`BitAnd`]
/// * [`BitOr`]
/// * [`BitXor`]
/// * [`Div`]
/// * [`Mul`]
/// * [`Rem`]
/// * [`Shl`]
/// * [`Shr`]
/// * [`Sub`]
///
/// [`Add`]: https://doc.rust-lang.org/std/ops/trait.Add.html
/// [`BitAnd`]: https://doc.rust-lang.org/std/ops/trait.BitAnd.html
/// [`BitOr`]: https://doc.rust-lang.org/std/ops/trait.BitOr.html
/// [`BitXor`]: https://doc.rust-lang.org/std/ops/trait.BitXor.html
/// [`Div`]: https://doc.rust-lang.org/std/ops/trait.Div.html
/// [`Mul`]: https://doc.rust-lang.org/std/ops/trait.Mul.html
/// [`Rem`]: https://doc.rust-lang.org/std/ops/trait.Rem.html
/// [`Shl`]: https://doc.rust-lang.org/std/ops/trait.Shl.html
/// [`Shr`]: https://doc.rust-lang.org/std/ops/trait.Shr.html
/// [`Sub`]: https://doc.rust-lang.org/std/ops/trait.Sub.html
#[doc(hidden)]
#[macro_export]
macro_rules! test_element_wise_binary_operators {
    () => {
        // Addition.
        $crate::test_element_wise_binary_operator_with_references!(
            element_wise_add,
            f64,
            [0.25, 1.33, -0.1, 1.0, -2.73, 1.2],
            [0.25, 1.33, -0.1, 1.0, -2.73, 1.2],
            +,
            [0.5, 2.66, -0.2, 2.0, -5.46, 2.4]
        );

        // Bitwise AND.
        $crate::test_element_wise_binary_operator_with_references!(
            element_wise_bit_and,
            u8,
            [7, 0, 1, 3, 5, 9],
            [4, 15, 1, 6, 5, 7],
            &,
            [4, 0, 1, 2, 5, 1]
        );

        // Bitwise OR.
        $crate::test_element_wise_binary_operator_with_references!(
            element_wise_bit_or,
            u8,
            [7, 0, 1, 3, 5, 9],
            [4, 15, 1, 6, 5, 7],
            |,
            [7, 15, 1, 7, 5, 15]
        );

        // Bitwise XOR.
        $crate::test_element_wise_binary_operator_with_references!(
            element_wise_bit_xor,
            u8,
            [7, 0, 1, 3, 5, 9],
            [4, 15, 1, 6, 5, 7],
            ^,
            [3, 15, 0, 5, 0, 14]
        );

        // Division.
        $crate::test_element_wise_binary_operator_with_references!(
            element_wise_div,
            f64,
            [1.0, 1.33, -0.1, 4.0, -2.73, 4.0],
            [2.0, 1.33, -4.0, -2.0, 2.73, 0.1],
            /,
            [0.5, 1.0, 0.025, -2.0, -1.0, 40.0]
        );

        // Multiplication.
        $crate::test_element_wise_binary_operator_with_references!(
            element_wise_mul,
            f64,
            [0.25, 1.0, -0.3, -1.0, 2.73, 1.2],
            [2.0, 0.25, -0.3, 2.0, -2.0, 1.2],
            *,
            [0.5, 0.25, 0.09, -2.0, -5.46, 1.44]
        );

        // Remainder.
        $crate::test_element_wise_binary_operator_with_references!(
            element_wise_rem,
            i64,
            [2, 6, -3, 5, 5, -10],
            [1, 4, 2, -4, 6, -2],
            %,
            [0, 2, -1, 1, 5, 0]
        );

        // Bitwise left shift.
        $crate::test_element_wise_binary_operator_with_references!(
            element_wise_shl,
            u8,
            [7, 0, 1, 5, 6, 3],
            [1, 5, 5, 0, 2, 3],
            <<,
            [14, 0, 32, 5, 24, 24]
        );

        // Bitwise right shift.
        $crate::test_element_wise_binary_operator_with_references!(
            element_wise_shr,
            u8,
            [7, 0, 1, 5, 6, 15],
            [1, 5, 5, 2, 2, 3],
            >>,
            [3, 0, 0, 1, 1, 1]
        );

        // Subtraction.
        $crate::test_element_wise_binary_operator_with_references!(
            element_wise_sub,
            f64,
            [0.25, 1.0, -0.1, 1.0, -2.73, 1.3],
            [0.25, 0.4, 0.1, -1.0, -2.0, 3.6],
            -,
            [0.0, 0.6, -0.2, 2.0, -0.73, -2.3]
        );
    };
}

/// Implement the tests for a given binary operator as an element-wise operation on a matrix,
/// for all combinations of owned and referenced applications of the operation.
///
/// Two tests will be implemented for each combination: one where the dimensions of the matrices
/// match and the operation succeeds, the other where the dimensions do not match and thus, the
/// operation fails.
///
/// # Parameters
///
/// * `$mod`: The name of the submodule in which the tests will be implemented.
/// * `$data_type`: The type of the data for the first and second matrix in the test.
/// * `$data_self`: The actual data array for the first (`self`) matrix in the test, must have a
///                 length of `6`.
/// * `$data_other`: The actual data array for the second (`other`) matrix in the test, must have a
///                  length of `6`.
/// * `$operator`: The operator of the element-wise binary operation.
/// * `$expected_result`: An array of expected values for the operation in the test.
///
/// # Example
///
/// Implement tests for the addition:
///
/// ```text
/// test_element_wise_binary_operator_with_references!(
///     add,
///     f64,
///     [0.0, 2.3, -1.2, 42.1337, 1.0, -4.4],
///     [0.1, 3.3, 2.2, -13.3742, -1.0, 4.8],
///     +,
///     [0.1, 5.6, 1.0, 55.5079, 0.0, 0.4]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! test_element_wise_binary_operator_with_references {
    ($mod:ident,
     $data_type:tt,
     $data_self:expr,
     $data_other:expr,
     $operator:tt,
     $expected_result:expr
    ) => {
        #[cfg(test)]
        mod $mod {
            use super::*;

            /// Owned to owned.
            $crate::test_element_wise_binary_operator!(
                owned_to_owned,
                $data_type,
                $data_self,
                $data_other,
                *,
                $operator,
                *,
                $expected_result
            );

            /// Owned to referenced.
            $crate::test_element_wise_binary_operator!(
                owned_to_referenced,
                $data_type,
                $data_self,
                $data_other,
                *,
                $operator,
                &,
                $expected_result
            );

            /// Referenced to owned.
            $crate::test_element_wise_binary_operator!(
                referenced_to_owned,
                $data_type,
                $data_self,
                $data_other,
                &,
                $operator,
                *,
                $expected_result
            );

            /// Referenced to referenced.
            $crate::test_element_wise_binary_operator!(
                referenced_to_referenced,
                $data_type,
                $data_self,
                $data_other,
                &,
                $operator,
                &,
                $expected_result
            );
        }
    };
}

/// Implement the tests for a given binary operator as an element-wise operation on a matrix.
///
/// Two tests will be implemented: one where the dimensions of the matrices match and the operation
/// succeeds, the other where the dimensions do not match and thus, the operation fails.
///
/// # Parameters
///
/// * `$mod`: The name of the submodule in which the tests will be implemented.
/// * `$data_type`: The type of the data for the first and second matrix in the test.
/// * `$data_self`: The actual data array for the first (`self`) matrix in the test, must have a
///                 length of `6`.
/// * `$data_other`: The actual data array for the second (`other`) matrix in the test, must have a
///                  length of `6`.
/// * `$lhs_access`: How to access the `self` matrix identifier, either `*` (by value) or `&`
///                  (by reference).
/// * `$operator`: The operator of the element-wise binary operation.
/// * `$rhs_access`: How to access the `other` matrix identifier, either `*` (by value) or `&`
///                  (by reference).
/// * `$expected_result`: An array of expected values for the operation in the test.
///
/// # Example
///
/// Implement tests for the addition of a `Matrix<T>` to which a `&'_ Matrix<T>` is added:
///
/// ```text
/// test_element_wise_binary_operator!(
///     owned_to_referenced,
///     f64,
///     [0.0, 2.3, -1.2, 42.1337, 1.0, -4.4],
///     [0.1, 3.3, 2.2, -13.3742, -1.0, 4.8],
///     *,
///     +,
///     &,
///     [0.1, 5.6, 1.0, 55.5079, 0.0, 0.4]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! test_element_wise_binary_operator {
    ($mod:ident,
     $data_type:tt,
     $data_self:expr,
     $data_other:expr,
     $lhs_access:tt,
     $operator:tt,
     $rhs_access:tt,
     $expected_result:expr
    ) => {
        #[cfg(test)]
        mod $mod {
            use super::*;

            /// Test the binary operator when the dimensions of both matrices match.
            #[test]
            fn correct_dimensions() {
                let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
                let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
                let data_self: [$data_type; 6] = $data_self;
                let data_other: [$data_type; 6] = $data_other;
                let matrix = Matrix::from_slice(rows, columns, &data_self).unwrap();
                let other  = Matrix::from_slice(rows, columns, &data_other).unwrap();

                let result = $crate::access_variable!($lhs_access matrix) $operator
                             $crate::access_variable!($rhs_access other);
                assert!(result.is_ok());
                assert_eq!(result.unwrap().as_slice(), $expected_result);
            }

            /// Test the binary operator when the dimensions of both matrices do not match.
            #[test]
            fn incorrect_dimensions() {
                let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
                let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
                let data_self: [$data_type; 6] = $data_self;
                let data_other: [$data_type; 6] = $data_other;
                let matrix = Matrix::from_slice(rows, columns, &data_self).unwrap();
                let other = Matrix::from_slice(columns, rows, &data_other).unwrap();

                let result = $crate::access_variable!($lhs_access matrix) $operator
                             $crate::access_variable!($rhs_access other);
                assert!(result.is_err());

                let is_correct_error: bool = match result.unwrap_err() {
                    Error::DimensionMismatch => true,
                    _ => false,
                };

                assert!(is_correct_error, "Expected error Error::DimensionMismatch not satisfied.");
            }
        }
    };
}

// endregion

// region Documentation

/// Get a documentation string for the element-wise binary operators.
///
/// # Parameters
///
/// * `$explanation`: A short explanation of what the operator does.
/// * `$data_type`: The type of the data for the first and second matrix in the example.
/// * `$data_self`: The actual data array for the first (`self`) matrix in the example. It must have
///                 a length of `6`.
/// * `$data_other`: The actual data array for the second (`other`) matrix in the example. It must
///                  have a length of `6`.
/// * `$lhs_ident`: The `self` matrix identifier, either `matrix` or `&matrix`.
/// * `$operator`: The operator of the element-wise binary operation.
/// * `$rhs_ident`: The `other` matrix identifier, either `other` or `&other`.
/// * `$expected_result`: An array of expected values for the operation in the example.
///
/// # Example
///
/// Get the documentation for element-wise addition:
///
/// ```text
/// doc_element_wise_binary_operator!(
///     "Element-wise add the values of `other` to `self`.",
///     f64,
///     [0.1, -2.33, 1.0, 3.3, 0.0, 42.1337],
///     [0.1, -2.33, 1.0, 3.3, 0.0, 42.1337],
///     matrix,
///     +,
///     &other,
///     [0.2, -4.66, 2.0, 6.6, 0.0, 84.2674]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! doc_element_wise_binary_operator {
    ($explanation:expr,
     $data_type:tt,
     $data_self:expr,
     $data_other:expr,
     $lhs_ident:expr,
     $operator:tt,
     $rhs_ident:expr,
     $expected_result:expr
    ) => {
        concat!(
            $explanation,
            "\n\n",
            "The dimensions of both matrices must match, otherwise an [`Error::DimensionMismatch`] \
             will be returned.",
            "\n\n",
            "# Example",
            "\n\n",
            "```\n",
            "use std::num::NonZeroUsize;\n",
            "use reural_network::matrix::Matrix;",
            "\n\n",
            "let rows = NonZeroUsize::new(2).unwrap();\n",
            "let columns = NonZeroUsize::new(3).unwrap();\n",
            "let data_matrix: [",
            stringify!($data_type),
            "; 6] = ",
            stringify!($data_self),
            ";\n",
            "let data_other: [",
            stringify!($data_type),
            "; 6] = ",
            stringify!($data_other),
            ";\n",
            "let matrix = Matrix::from_slice(rows, columns, &data_matrix).unwrap();\n",
            "let other = Matrix::from_slice(rows, columns, &data_other).unwrap();",
            "\n\n",
            "let result = ",
            stringify!($lhs_ident),
            " ",
            stringify!($operator),
            " ",
            stringify!($rhs_ident),
            ";\n",
            "assert_eq!(result.unwrap().as_slice(), &",
            stringify!($expected_result),
            ");\n",
            "```",
            "\n\n",
            "[`Error::DimensionMismatch`]: ../enum.Error.html#variant.DimensionMismatch"
        );
    };
}

// endregion
