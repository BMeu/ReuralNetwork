// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Macros to implement unary operations.
//!
//! The main macros in this module are [`impl_unary_operators`] to implement all unary operations,
//!  and [`test_unary_operators`] to test these implementations.
//!
//! [`impl_unary_operators`]: ../../macro.impl_unary_operators.html
//! [`test_unary_operators`]: ../../macro.test_unary_operators.html

// region Implement

/// Implement all unary operators on a matrix `Matrix<T>` and on `&'_ Matrix<T>`.
///
/// # Implemented Unary Operators Traits
///
/// * [`Neg`]
/// * [`Not`]
///
/// All these traits must be `use`d in the module calling the macro.
///
/// [`Neg`]: https://doc.rust-lang.org/std/ops/trait.Neg.html
/// [`Not`]: https://doc.rust-lang.org/std/ops/trait.Not.html
#[doc(hidden)]
#[macro_export]
macro_rules! impl_unary_operators {
    () => {
        // Negation.
        $crate::impl_unary_operator_with_references!(
            Neg,
            neg,
            -,
            "Negate all elements in `self`.",
            f64,
            [0.25, 1.33, -0.1, 0.0, -2.73, 1.2],
            [-0.25, -1.33, 0.1, 0.0, 2.73, -1.2]
        );

        // Logical Negation.
        $crate::impl_unary_operator_with_references!(
            Not,
            not,
            !,
            "Logically negate all elements in `self`.",
            bool,
            [true, false, false, false, true, false],
            [false, true, true, true, false, true]
        );
    };
}

/// Implement a given unary operator on `Matrix<T>` and on `&'_ Matrix<T>`.
///
/// # Parameters
///
/// * `$trait`: The unary-operator trait to implement. This trait must also be implemented by `T`.
/// * `$fn`: The name of the function that implements the unary operator.
/// * `$operator`: The actual unary operator, e.g. `-` for the `Neg` trait.
/// * `$explanation`: A short explanation for the documentation of what the operator does.
/// * `$data_type`: The type `T` of the data in the matrix in the documentation example.
/// * `$data`: The actual data array for the matrix in the documentation example. It must have a
///            length of `6`.
/// * `$expected_result`: An array of expected values for the operation in the documentation
///                       example.
///
/// # Example
///
/// Implement negation:
///
/// ```text
/// impl_unary_operator_with_references!(
///     Neg,
///     neg,
///     -,
///     "Negate all elements in `self`",
///     f64,
///     [0.25, 1.33, -0.1, 1.0, -2.73, 1.2],
///     [-0.25, -1.33, 0.1, -1.0, 2.73, -1.2],
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! impl_unary_operator_with_references {
    ($trait:tt,
     $fn:tt,
     $operator:tt,
     $explanation:expr,
     $data_type:ty,
     $data:expr,
     $result:expr
    ) => {
        // Implement the operator for Matrix<T>.
        $crate::impl_unary_operator!(
            *,
            $trait,
            $fn,
            $operator,
            $crate::doc_unary_operator!(
                $explanation,
                $data_type,
                $data,
                *,
                $operator,
                $result
            )
        );

        // Implement the operator for &'_ Matrix<T>.
        $crate::impl_unary_operator!(
            &,
            $trait,
            $fn,
            $operator,
            $crate::doc_unary_operator!(
                $explanation,
                $data_type,
                $data,
                &,
                $operator,
                $result
            )
        );
    };
}

/// Implement a given unary operator on a matrix whose element type also implements the operator.
///
/// # Parameters
///
/// * `$access`: The access type of the operator, either `*` for owned access or `&` for referenced
///              access.
/// * `$trait`: The unary-operator trait to implement. This trait must also be implemented by `T`.
/// * `$fn`: The name of the function that implements the unary operator.
/// * `$operator`: The actual binary operator, e.g. `-` for the `Neg` trait.
/// * `$documentation`: The documentation for the operator method.
///
/// # Example
///
/// Implement negation for `Matrix<T>`:
///
/// ```text
/// impl_unary_operator!(
///     *,
///     Neg,
///     neg,
///     -,
///     "Negate all elements in `self`."
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! impl_unary_operator {
    ($access:tt, $trait:tt, $fn:tt, $operator:tt, $documentation:expr) => {
        impl<T> $trait for $crate::specify_matrix_type!($access)
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = Matrix<T>;

            #[doc = $documentation]
            fn $fn(self) -> Self::Output {
                let mut result: Matrix<T> = Matrix {
                    rows: self.rows,
                    columns: self.columns,
                    data: self.data.clone(),
                };

                result.map(|element, _row, _column| $operator element);

                result
            }
        }
    };
}

// endregion

// region Tests

/// Implement tests for all unary operations on a matrix `Matrix<T>`.
///
/// # Tested Binary Operators Traits
///
/// * [`Neg`]
/// * [`Not`]
///
/// [`Neg`]: https://doc.rust-lang.org/std/ops/trait.Neg.html
/// [`Not`]: https://doc.rust-lang.org/std/ops/trait.Not.html
#[doc(hidden)]
#[macro_export]
macro_rules! test_unary_operators {
    () => {
        // Negation.
        $crate::test_unary_operator_with_references!(
            neg,
            f64,
            [0.25, 1.33, -0.1, 0.0, -2.73, 1.2],
            -,
            [-0.25, -1.33, 0.1, 0.0, 2.73, -1.2]
        );

        // Logical negation.
        $crate::test_unary_operator_with_references!(
            not,
            bool,
            [true, true, false, false, true, false],
            !,
            [false, false, true, true, false, true]
        );
    };
}

/// Implement the tests for a given unary operator on both an owned and a referenced matrix.
///
/// # Parameters
///
/// * `$mod`: The name of the submodule in which the tests will be implemented.
/// * `$data_type`: The type `T` of the data in the matrix in the test.
/// * `$data`: The actual data array for the matrix in the test, must have a length of `6`.
/// * `$operator`: The operator of the unary operation.
/// * `$expected_result`: An array of expected values for the operation in the test.
///
/// # Example
///
/// Implement tests for the negation:
///
/// ```text
/// test_unary_operator_with_references!(
///     neg,
///     f64,
///     [0.0, 2.3, -1.2, 42.1337, 1.0, -4.4],
///     -,
///     [0.0, -2.3, 1.2, -42.1337, -1.0, 4.4]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! test_unary_operator_with_references {
    ($mod:ident,
     $data_type:tt,
     $data:expr,
     $operator:tt,
     $expected_result:expr
    ) => {
        #[cfg(test)]
        mod $mod {
            use super::*;

            /// Owned.
            $crate::test_unary_operator!(
                owned,
                $data_type,
                $data,
                *,
                $operator,
                $expected_result
            );

            /// Referenced.
            $crate::test_unary_operator!(
                referenced,
                $data_type,
                $data,
                &,
                $operator,
                $expected_result
            );
        }
    };
}

/// Implement the tests for a given unary operator on a matrix.
///
/// # Parameters
///
/// * `$mod`: The name of the submodule in which the tests will be implemented.
/// * `$data_type`: The type `T` of the data in the matrix in the test.
/// * `$data`: The actual data array for the matrix in the test, must have a length of `6`.
/// * `$access`: How to access the `self` matrix identifier, either `*` (by value) or `&` (by
///              reference).
/// * `$operator`: The operator of the unary operation.
/// * `$expected_result`: An array of expected values for the operation in the test.
///
/// # Example
///
/// Implement tests for the negation of a `Matrix<T>`:
///
/// ```text
/// test_unary_operator!(
///     owned,
///     f64,
///     [0.0, 2.3, -1.2, 42.1337, 1.0, -4.4],
///     *,
///     !,
///     [0.0, -2.3, 1.2, -42.1337, -1.0, 4.4]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! test_unary_operator {
    ($mod:ident,
     $data_type:tt,
     $data:expr,
     $access:tt,
     $operator:tt,
     $expected_result:expr
    ) => {
        #[cfg(test)]
        mod $mod {
            use super::*;

            /// Test the unary operator.
            #[test]
            fn correct_dimensions() {
                let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
                let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
                let data: [$data_type; 6] = $data;
                let matrix = Matrix::from_slice(rows, columns, &data).unwrap();

                let result = $operator$crate::access_variable!($access matrix);
                assert_eq!(result.as_slice(), $expected_result);
            }
        }
    };
}

// endregion

// region Documentation

/// Get a documentation string for the unary operators.
///
/// # Parameters
///
/// * `$explanation`: A short explanation of what the operator does.
/// * `$data_type`: The type `T` of the data in the matrix in the example.
/// * `$data`: The actual data array for the matrix in the example. It must have a length of `6`.
/// * `$access`: How to access the `self` matrix identifier, either `*` (by value) or `&` (by
///              reference).
/// * `$operator`: The operator of the scalar binary operation.
/// * `$expected_result`: An array of expected values for the operation in the example.
///
/// # Example
///
/// Get the documentation for negation:
///
/// ```text
/// doc_unary_operator!(
///     "Negate all elements in `self`.",
///     f64,
///     [0.1, -2.33, 1.0, 3.3, 0.0, 42.1337],
///     *,
///     -,
///     [-0.1, 2.33, -1.0, -3.3, 0.0, -42.1337]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! doc_unary_operator {
    ($explanation:expr,
     $data_type:tt,
     $data:expr,
     $access:tt,
     $operator:tt,
     $expected_result:expr
    ) => {
        concat!(
            $explanation,
            "\n\n",
            "# Example",
            "\n\n",
            "```\n",
            "use std::num::NonZeroUsize;\n",
            "use reural_network::matrix::Matrix;",
            "\n\n",
            "let rows = NonZeroUsize::new(2).unwrap();\n",
            "let columns = NonZeroUsize::new(3).unwrap();\n",
            "let data: [",
            stringify!($data_type),
            "; 6] = ",
            stringify!($data),
            ";\n",
            "let matrix = Matrix::from_slice(rows, columns, &data).unwrap();",
            "\n\n",
            "let result = ",
            stringify!($operator),
            $crate::access_variable_as_string!($access matrix),
            ";\n",
            "assert_eq!(result.as_slice(), &",
            stringify!($expected_result),
            ");\n",
            "```"
        );
    };
}

// endregion
