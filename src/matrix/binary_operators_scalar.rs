// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Macros to implement scalar binary operations.
//!
//! The main macros in this module are [`impl_scalar_binary_operators`] to implement all
//! binary operators as scalar operations, and [`test_scalar_binary_operators`] to test
//! these implementations.
//!
//! [`impl_scalar_binary_operators`]: ../../macro.impl_scalar_binary_operators.html
//! [`test_scalar_binary_operators`]: ../../macro.test_scalar_binary_operators.html

// region Implement

/// Implement all binary operators as scalar operations on a matrix `Matrix<T>` and a scalar value
/// `T` and all possible combinations including (immutable) references of these types.
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
macro_rules! impl_scalar_binary_operators {
    () => {
        // Addition.
        $crate::impl_scalar_binary_operator_with_references!(
            Add,
            add,
            +,
            "Add `other` to all elements in `self`.",
            f64,
            [0.25, 1.33, -0.1, 1.0, -2.73, 1.2],
            1.3,
            [1.55, 2.63, 1.2, 2.3, -1.43, 2.5]
        );

        // Bitwise AND.
        $crate::impl_scalar_binary_operator_with_references!(
            BitAnd,
            bitand,
            &,
            "Calculate the bitwise AND of each element in `self` with `other`.",
            u8,
            [7, 0, 1, 3, 5, 9],
            4,
            [4, 0, 0, 0, 4, 0]
        );

        // Bitwise OR.
        $crate::impl_scalar_binary_operator_with_references!(
            BitOr,
            bitor,
            |,
            "Calculate the bitwise OR of each element in `self` with `other`.",
            u8,
            [7, 0, 1, 3, 5, 9],
            4,
            [7, 4, 5, 7, 5, 13]
        );

        // Bitwise XOR.
        $crate::impl_scalar_binary_operator_with_references!(
            BitXor,
            bitxor,
            ^,
            "Calculate the bitwise XOR of each element in `self` with `other`.",
            u8,
            [7, 0, 1, 3, 5, 9],
            4,
            [3, 4, 5, 7, 1, 13]
        );

        // Division.
        $crate::impl_scalar_binary_operator_with_references!(
            Div,
            div,
            /,
            "Divide each element in `self` by `other`.",
            f64,
            [1.0, 1.33, -0.1, 4.0, -2.73, 80.0],
            2.0,
            [0.5, 0.665, -0.05, 2.0, -1.365, 40.0]
        );

        // Multiplication.
        $crate::impl_scalar_binary_operator_with_references!(
            Mul,
            mul,
            *,
            "Multiply each element in `self` by `other`.",
            f64,
            [0.25, 1.0, -0.3, -1.0, 2.73, 1.2],
            2.0,
            [0.5, 2.0, -0.6, -2.0, 5.46, 2.4]
        );

        // Remainder.
        $crate::impl_scalar_binary_operator_with_references!(
            Rem,
            rem,
            %,
            "Calculate the remainder when dividing each element in `self` by `other`.",
            i64,
            [2, 6, -3, 5, -5, -10],
            4,
            [2, 2, -3, 1, -1, -2]
        );

        // Bitwise left shift.
        $crate::impl_scalar_binary_operator_with_references!(
            Shl,
            shl,
            <<,
            "Bitwise shift each element in `self` to the left by `other`.",
            u8,
            [7, 0, 1, 5, 6, 3],
            2,
            [28, 0, 4, 20, 24, 12]
        );

        // Bitwise right shift.
        $crate::impl_scalar_binary_operator_with_references!(
            Shr,
            shr,
            >>,
            "Bitwise shift each element in `self` to the right by `other`.",
            u8,
            [7, 0, 1, 5, 6, 15],
            1,
            [3, 0, 0, 2, 3, 7]
        );

        // Subtraction.
        $crate::impl_scalar_binary_operator_with_references!(
            Sub,
            sub,
            -,
            "Subtract `other` from all elements in `self`.",
            f64,
            [0.25, 1.0, -0.1, -1.0, -2.73, 1.3],
            0.25,
            [0.0, 0.75, -0.35, -1.25, -2.98, 1.05]
        );
    };
}

/// Implement a given binary operator as a scalar operation on `Matrix<T>` with a scalar `T` and on
/// `&'_ Matrix<T>` with a scalar `T`.
///
/// # Parameters
///
/// * `$trait`: The binary-operator trait to implement. This trait must also be implemented by `T`.
/// * `$fn`: The name of the function that implements the binary operator.
/// * `$operator`: The actual binary operator, e.g. `+` for the `Add` trait.
/// * `$explanation`: A short explanation for the documentation of what the operator does.
/// * `$data_type`: The type `T` of the data in the matrix in the documentation example.
/// * `$data_self`: The actual data array for the matrix in the documentation example. It must have
///                 a length of `6`.
/// * `$data_other`: The scalar value added to the matrix in the documentation example.
/// * `$expected_result`: An array of expected values for the operation in the documentation
///                       example.
///
/// # Example
///
/// Implement addition::
///
/// ```text
/// impl_scalar_binary_operator_with_references!(
///     Add,
///     add,
///     +,
///     "Add `other` to all elements in `self`",
///     f64,
///     [0.25, 1.33, -0.1, 1.0, -2.73, 1.2],
///     1.3,
///     [1.55, 2.63, 1.2, 2.3, -1.43, 2.5]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! impl_scalar_binary_operator_with_references {
    ($trait:tt,
     $fn:tt,
     $operator:tt,
     $explanation:expr,
     $data_type:ty,
     $data_self:expr,
     $data_other:expr,
     $result:expr
    ) => {
        // Implement the operator for Matrix<T> and T.
        $crate::impl_scalar_binary_operator!(
            *,
            $trait,
            $fn,
            $operator,
            $crate::doc_scalar_binary_operator!(
                $explanation,
                $data_type,
                $data_self,
                $data_other,
                *,
                $operator,
                $result
            )
        );

        // Implement the operator for &'_ Matrix<T> and T.
        $crate::impl_scalar_binary_operator!(
            &,
            $trait,
            $fn,
            $operator,
            $crate::doc_scalar_binary_operator!(
                $explanation,
                $data_type,
                $data_self,
                $data_other,
                &,
                $operator,
                $result
            )
        );
    };
}

/// Implement a given binary operator as a scalar operation on a matrix whose element type
/// also implements the operator and a scalar value.
///
/// # Parameters
///
/// * `$access`: The left-hand side access type of the operator, either `*` for owned access or `&`
///              for referenced access.
/// * `$trait`: The binary-operator trait to implement. This trait must also be implemented by `T`.
/// * `$fn`: The name of the function that implements the binary operator.
/// * `$operator`: The actual binary operator, e.g. `+` for the `Add` trait.
/// * `$documentation`: The documentation for the operator method.
///
/// # Example
///
/// Implement addition for `Matrix<T>` to which a `T` is added:
///
/// ```text
/// impl_scalar_binary_operator!(
///     *,
///     Add,
///     add,
///     +,
///     "Add `other` to all elements in `self`."
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! impl_scalar_binary_operator {
    ($access:tt, $trait:tt, $fn:tt, $operator:tt, $documentation:expr) => {
        impl<T> $trait<T> for $crate::specify_matrix_type!($access)
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = Matrix<T>;

            #[doc = $documentation]
            fn $fn(self, other: T) -> Self::Output {
                let mut result: Matrix<T> = Matrix {
                    rows: self.rows,
                    columns: self.columns,
                    data: self.data.clone(),
                };

                result.map(|element, _row, _column| element $operator other);

                result
            }
        }
    };
}

// endregion

// region Tests

/// Implement tests for all scalar binary operations on a matrix `Matrix<T>` and a scalar value `T`.
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
macro_rules! test_scalar_binary_operators {
    () => {
        // Addition.
        $crate::test_scalar_binary_operator_with_references!(
            scalar_add,
            i8,
            [7, 5, -6, 0, 3, 1],
            42,
            +,
            [49, 47, 36, 42, 45, 43]
        );

        // Bitwise AND.
        $crate::test_scalar_binary_operator_with_references!(
            scalar_bit_and,
            u8,
            [7, 0, 1, 3, 5, 9],
            4,
            &,
            [4, 0, 0, 0, 4, 0]
        );

        // Bitwise OR.
        $crate::test_scalar_binary_operator_with_references!(
            scalar_bit_or,
            u8,
            [7, 0, 1, 3, 5, 9],
            4,
            |,
            [7, 4, 5, 7, 5, 13]
        );

        // Bitwise XOR.
        $crate::test_scalar_binary_operator_with_references!(
            scalar_bit_xor,
            u8,
            [7, 0, 1, 3, 5, 9],
            4,
            ^,
            [3, 4, 5, 7, 1, 13]
        );

        // Division.
        $crate::test_scalar_binary_operator_with_references!(
            scalar_div,
            i8,
            [4, 2, -6, 4, -10, 80],
            2,
            /,
            [2, 1, -3, 2, -5, 40]
        );

        // Multiplication.
        $crate::test_scalar_binary_operator_with_references!(
            scalar_mul,
            i64,
            [25, 1, -3, -1, 2, 1],
            2,
            *,
            [50, 2, -6, -2, 4, 2]
        );

        // Remainder.
        $crate::test_scalar_binary_operator_with_references!(
            scalar_rem,
            i64,
            [2, 6, -3, 5, -5, -10],
            4,
            %,
            [2, 2, -3, 1, -1, -2]
        );

        // Bitwise left shift.
        $crate::test_scalar_binary_operator_with_references!(
            scalar_shl,
            u8,
            [7, 0, 1, 5, 6, 3],
            2,
            <<,
            [28, 0, 4, 20, 24, 12]
        );

        // Bitwise right shift.
        $crate::test_scalar_binary_operator_with_references!(
            scalar_shr,
            u8,
            [7, 0, 1, 5, 6, 15],
            1,
            >>,
            [3, 0, 0, 2, 3, 7]
        );

        // Subtraction.
        $crate::test_scalar_binary_operator_with_references!(
            scalar_sub,
            i8,
            [5, 1, -10, -2, 25, 13],
            3,
            -,
            [2, -2, -13, -5, 22, 10]
        );
    };
}

/// Implement the tests for a given binary operator as an scalar operation on a matrix and a scalar
/// value, for all combinations of owned and referenced applications of the operation.
///
/// # Parameters
///
/// * `$mod`: The name of the submodule in which the tests will be implemented.
/// * `$data_type`: The type `T` of the data in the matrix in the test.
/// * `$data_self`: The actual data array for the matrix in the test, must have a length of `6`.
/// * `$data_other`: The scalar value `other` in the test.
/// * `$operator`: The operator of the scalar binary operation.
/// * `$expected_result`: An array of expected values for the operation in the test.
///
/// # Example
///
/// Implement tests for the addition:
///
/// ```text
/// test_scalar_binary_operator_with_references!(
///     add,
///     f64,
///     [0.0, 2.3, -1.2, 42.1337, 1.0, -4.4],
///     0.1,
///     +,
///     [0.1, 2.4, -1.1, 42.2337, 1.1, -4.3]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! test_scalar_binary_operator_with_references {
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

            // Owned to owned.
            $crate::test_scalar_binary_operator!(
                owned,
                $data_type,
                $data_self,
                $data_other,
                *,
                $operator,
                $expected_result
            );

            // Referenced to owned.
            $crate::test_scalar_binary_operator!(
                referenced,
                $data_type,
                $data_self,
                $data_other,
                &,
                $operator,
                $expected_result
            );
        }
    };
}

/// Implement the tests for a given binary operator as a scalar operation on a matrix and a scalar
/// value.
///
/// # Parameters
///
/// * `$mod`: The name of the submodule in which the tests will be implemented.
/// * `$data_type`: The type `T` of the data in the matrix in the test.
/// * `$data_self`: The actual data array for the matrix in the test, must have a length of `6`.
/// * `$data_other`: The scalar value of `other`.
/// * `$access`: How to access the `self` matrix identifier, either `*` (by value) or `&` (by
///              reference).
/// * `$operator`: The operator of the scalar binary operation.
/// * `$expected_result`: An array of expected values for the operation in the test.
///
/// # Example
///
/// Implement tests for the addition of a `Matrix<T>` to which a `&'_ T` is added:
///
/// ```text
/// test_scalar_binary_operator!(
///     own,
///     f64,
///     [0.0, 2.3, -1.2, 42.1337, 1.0, -4.4],
///     0.1,
///     *,
///     +,
///     [0.1, 2.4, -1.1, 42.2337, 1.1, -4.3]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! test_scalar_binary_operator {
    ($mod:ident,
     $data_type:tt,
     $data_self:expr,
     $data_other:expr,
     $access:tt,
     $operator:tt,
     $expected_result:expr
    ) => {
        #[cfg(test)]
        mod $mod {
            use super::*;

            /// Test the binary operator.
            #[test]
            fn correct_dimensions() {
                let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
                let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
                let data_self: [$data_type; 6] = $data_self;
                let other: $data_type = $data_other;
                let matrix = Matrix::from_slice(rows, columns, &data_self).unwrap();

                let result = $crate::access_variable!($access matrix) $operator other;
                assert_eq!(result.as_slice(), $expected_result);
            }
        }
    };
}

// endregion

// region Documentation

/// Get a documentation string for the scalar binary operators.
///
/// # Parameters
///
/// * `$explanation`: A short explanation of what the operator does.
/// * `$data_type`: The type `T` of the data in the matrix in the example.
/// * `$data_self`: The actual data array for the matrix in the example. It must have a length of
///                 `6`.
/// * `$data_other`: The scalar value added to the matrix in the example.
/// * `$access`: How to access the `self` matrix identifier, either `*` (by value) or `&` (by
///              reference).
/// * `$operator`: The operator of the scalar binary operation.
/// * `$expected_result`: An array of expected values for the operation in the example.
///
/// # Example
///
/// Get the documentation for scalar addition:
///
/// ```text
/// doc_scalar_binary_operator!(
///     "Add `other` to all elements in `self`.",
///     f64,
///     [0.1, -2.33, 1.0, 3.3, 0.0, 42.1337],
///     1.3,
///     *,
///     +,
///     [1.4, -1.03, 2.3, 4.6, 1.3, 43.4337]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! doc_scalar_binary_operator {
    ($explanation:expr,
     $data_type:tt,
     $data_self:expr,
     $data_other:expr,
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
            "let data_matrix: [",
            stringify!($data_type),
            "; 6] = ",
            stringify!($data_self),
            ";\n",
            "let other: ",
            stringify!($data_type),
            " = ",
            stringify!($data_other),
            ";\n",
            "let matrix = Matrix::from_slice(rows, columns, &data_matrix).unwrap();",
            "\n\n",
            "let result = ",
            $crate::access_variable_as_string!($access matrix),
            " ",
            stringify!($operator),
            " other;\n",
            "assert_eq!(result.as_slice(), &",
            stringify!($expected_result),
            ");\n",
            "```"
        );
    };
}

// endregion
