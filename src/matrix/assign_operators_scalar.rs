// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT o
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Macros to implement scalar assign operations.
//!
//! The main macros in this module are [`impl_scalar_assign_operators`] to implement all
//! assign operators as scalar operations, and [`test_scalar_assign_operators`] to test
//! these implementations.
//!
//! [`impl_scalar_assign_operators`]: ../../macro.impl_scalar_assign_operators.html
//! [`test_scalar_assign_operators`]: ../../macro.test_scalar_assign_operators.html

// region Implement

/// Implement all assign operators as scalar operations on a matrix `Matrix<T>` and a scalar value
/// `T`.
///
/// # Implemented Binary Operators Traits
///
/// * [`AddAssign`]
/// * [`BitAndAssign`]
/// * [`BitOrAssign`]
/// * [`BitXorAssign`]
/// * [`DivAssign`]
/// * [`MulAssign`]
/// * [`RemAssign`]
/// * [`ShlAssign`]
/// * [`ShrAssign`]
/// * [`SubAssign`]
///
/// All these traits must be `use`d in the module calling the macro.
///
/// [`AddAssign`]: https://doc.rust-lang.org/std/ops/trait.AddAssign.html
/// [`BitAndAssign`]: https://doc.rust-lang.org/std/ops/trait.BitAndAssign.html
/// [`BitOrAssign`]: https://doc.rust-lang.org/std/ops/trait.BitOrAssign.html
/// [`BitXorAssign`]: https://doc.rust-lang.org/std/ops/trait.BitXorAssign.html
/// [`DivAssign`]: https://doc.rust-lang.org/std/ops/trait.DivAssign.html
/// [`MulAssign`]: https://doc.rust-lang.org/std/ops/trait.MulAssign.html
/// [`RemAssign`]: https://doc.rust-lang.org/std/ops/trait.RemAssign.html
/// [`ShlAssign`]: https://doc.rust-lang.org/std/ops/trait.ShlAssign.html
/// [`ShrAssign`]: https://doc.rust-lang.org/std/ops/trait.ShrAssign.html
/// [`SubAssign`]: https://doc.rust-lang.org/std/ops/trait.SubAssign.html
#[doc(hidden)]
#[macro_export]
macro_rules! impl_scalar_assign_operators {
    () => {
        // Addition.
        $crate::impl_scalar_assign_operator!(
            AddAssign,
            add_assign,
            +=,
            $crate::doc_scalar_assign_operator!(
                "Add `other` to all elements in `self`.",
                i64,
                [25, 133, -1, 1, -273, 12],
                13,
                +=,
                [38, 146, 12, 14, -260, 25]
            )
        );

        // Bitwise AND.
        $crate::impl_scalar_assign_operator!(
            BitAndAssign,
            bitand_assign,
            &=,
            $crate::doc_scalar_assign_operator!(
                "Calculate the bitwise AND of each element in `self` with `other`.",
                u8,
                [7, 0, 1, 3, 5, 9],
                4,
                &=,
                [4, 0, 0, 0, 4, 0]
            )
        );

        // Bitwise OR.
        $crate::impl_scalar_assign_operator!(
            BitOrAssign,
            bitor_assign,
            |=,
            $crate::doc_scalar_assign_operator!(
                "Calculate the bitwise OR of each element in `self` with `other`.",
                u8,
                [7, 0, 1, 3, 5, 9],
                4,
                |=,
                [7, 4, 5, 7, 5, 13]
            )
        );

        // Bitwise XOR.
        $crate::impl_scalar_assign_operator!(
            BitXorAssign,
            bitxor_assign,
            ^=,
            $crate::doc_scalar_assign_operator!(
                "Calculate the bitwise XOR of each element in `self` with `other`.",
                u8,
                [7, 0, 1, 3, 5, 9],
                4,
                ^=,
                [3, 4, 5, 7, 1, 13]
            )
        );

        // Division.
        $crate::impl_scalar_assign_operator!(
            DivAssign,
            div_assign,
            /=,
            $crate::doc_scalar_assign_operator!(
                "Divide each element in `self` by `other`.",
                i64,
                [10, 130, -10, 4, -46, 0],
                2,
                /=,
                [5, 65, -5, 2, -23, 0]
            )
        );

        // Multiplication.
        $crate::impl_scalar_assign_operator!(
            MulAssign,
            mul_assign,
            *=,
            $crate::doc_scalar_assign_operator!(
                "Multiply each element in `self` by `other`.",
                i64,
                [25, 10, -3, -1, 0, 12],
                2,
                *=,
                [50, 20, -6, -2, 0, 24]
            )
        );

        // Remainder.
        $crate::impl_scalar_assign_operator!(
            RemAssign,
            rem_assign,
            %=,
            $crate::doc_scalar_assign_operator!(
                "Calculate the remainder when dividing each element in `self` by `other`.",
                i64,
                [2, 6, -3, 5, -5, -10],
                4,
                %=,
                [2, 2, -3, 1, -1, -2]
            )
        );

        // Bitwise left shift.
        $crate::impl_scalar_assign_operator!(
            ShlAssign,
            shl_assign,
            <<=,
            $crate::doc_scalar_assign_operator!(
                "Bitwise shift each element in `self` to the left by `other`.",
                u8,
                [7, 0, 1, 5, 6, 3],
                2,
                <<=,
                [28, 0, 4, 20, 24, 12]
            )
        );

        // Bitwise right shift.
        $crate::impl_scalar_assign_operator!(
            ShrAssign,
            shr_assign,
            >>=,
            $crate::doc_scalar_assign_operator!(
                "Bitwise shift each element in `self` to the right by `other`.",
                u8,
                [7, 0, 1, 5, 6, 15],
                1,
                >>=,
                [3, 0, 0, 2, 3, 7]
            )
        );

        // Subtraction.
        $crate::impl_scalar_assign_operator!(
            SubAssign,
            sub_assign,
            -=,
            $crate::doc_scalar_assign_operator!(
                "Subtract `other` from all elements in `self`.",
                i64,
                [25, 1, -25, 0, -273, 13],
                25,
                -=,
                [0, -24, -50, -25, -298, -12]
            )
        );
    };
}

/// Implement a given assign operator as a scalar operation on a matrix whose element type also
/// implements the operator and a scalar value.
///
/// # Parameters
///
/// * `$trait`: The assign operator trait to implement. This trait must also be implemented by `T`.
/// * `$fn`: The name of the function that implements the assign operator.
/// * `$operator`: The actual assign operator, e.g. `+=` for the `AddAssign` trait.
/// * `$documentation`: The documentation for the operator method.
///
/// # Example
///
/// Implement addition for `Matrix<T>` to which a `T` is added:
///
/// ```text
/// impl_scalar_assign_operator!(
///     AddAssign,
///     add_assign,
///     +=,
///     "Add `other` to all elements in `self`."
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! impl_scalar_assign_operator {
    ($trait:tt, $fn:tt, $operator:tt, $documentation:expr) => {
        impl<T> $trait<T> for $crate::specify_matrix_type!(*)
        where
            T: $trait<T> + Copy,
        {
            #[doc = $documentation]
            fn $fn(&mut self, other: T) {
                self.map_ref_mut(|element, _row, _column| *element $operator other);
            }
        }
    };
}

// endregion

// region Tests

/// Implement tests for all scalar assign operations on a matrix `Matrix<T>` and a scalar value `T`.
///
/// # Tested Assign Operators Traits
///
/// * [`AddAssign`]
/// * [`BitAndAssign`]
/// * [`BitOrAssign`]
/// * [`BitXorAssign`]
/// * [`DivAssign`]
/// * [`MulAssign`]
/// * [`RemAssign`]
/// * [`ShlAssign`]
/// * [`ShrAssign`]
/// * [`SubAssign`]
///
/// [`AddAssign`]: https://doc.rust-lang.org/std/ops/trait.AddAssign.html
/// [`BitAndAssign`]: https://doc.rust-lang.org/std/ops/trait.BitAndAssign.html
/// [`BitOrAssign`]: https://doc.rust-lang.org/std/ops/trait.BitOrAssign.html
/// [`BitXorAssign`]: https://doc.rust-lang.org/std/ops/trait.BitXorAssign.html
/// [`DivAssign`]: https://doc.rust-lang.org/std/ops/trait.DivAssign.html
/// [`MulAssign`]: https://doc.rust-lang.org/std/ops/trait.MulAssign.html
/// [`RemAssign`]: https://doc.rust-lang.org/std/ops/trait.RemAssign.html
/// [`ShlAssign`]: https://doc.rust-lang.org/std/ops/trait.ShlAssign.html
/// [`ShrAssign`]: https://doc.rust-lang.org/std/ops/trait.ShrAssign.html
/// [`SubAssign`]: https://doc.rust-lang.org/std/ops/trait.SubAssign.html
#[doc(hidden)]
#[macro_export]
macro_rules! test_scalar_assign_operators {
    () => {
        // Addition.
        $crate::test_scalar_assign_operator!(
            scalar_add_assign,
            i64,
            [25, 133, -1, 1, -273, 12],
            13,
            +=,
            [38, 146, 12, 14, -260, 25]
        );

        // Bitwise AND.
        $crate::test_scalar_assign_operator!(
            scalar_bit_and_assign,
            u8,
            [7, 0, 1, 3, 5, 9],
            4,
            &=,
            [4, 0, 0, 0, 4, 0]
        );

        // Bitwise OR.
        $crate::test_scalar_assign_operator!(
            scalar_bit_or_assign,
            u8,
            [7, 0, 1, 3, 5, 9],
            4,
            |=,
            [7, 4, 5, 7, 5, 13]
        );

        // Bitwise XOR.
        $crate::test_scalar_assign_operator!(
            scalar_bit_xor_assign,
            u8,
            [7, 0, 1, 3, 5, 9],
            4,
            ^=,
            [3, 4, 5, 7, 1, 13]
        );

        // Division.
        $crate::test_scalar_assign_operator!(
            scalar_div_assign,
            i64,
            [10, 130, -10, 4, -46, 0],
            2,
            /=,
            [5, 65, -5, 2, -23, 0]
        );

        // Multiplication.
        $crate::test_scalar_assign_operator!(
            scalar_mul_assign,
            i64,
            [25, 10, -3, -1, 0, 12],
            2,
            *=,
            [50, 20, -6, -2, 0, 24]
        );

        // Remainder.
        $crate::test_scalar_assign_operator!(
            scalar_rem_assign,
            i64,
            [2, 6, -3, 5, -5, -10],
            4,
            %=,
            [2, 2, -3, 1, -1, -2]
        );

        // Bitwise left shift.
        $crate::test_scalar_assign_operator!(
            scalar_shl_assign,
            u8,
            [7, 0, 1, 5, 6, 3],
            2,
            <<=,
            [28, 0, 4, 20, 24, 12]
        );

        // Bitwise right shift.
        $crate::test_scalar_assign_operator!(
            scalar_shr_assign,
            u8,
            [7, 0, 1, 5, 6, 15],
            1,
            >>=,
            [3, 0, 0, 2, 3, 7]
        );

        // Subtraction.
        $crate::test_scalar_assign_operator!(
            scalar_sub_assign,
            i64,
            [25, 1, -25, 0, -273, 13],
            25,
            -=,
            [0, -24, -50, -25, -298, -12]
        );
    };
}

/// Implement the tests for a given assign operator as a scalar operation on a matrix and a scalar
/// value.
///
/// # Parameters
///
/// * `$mod`: The name of the submodule in which the tests will be implemented.
/// * `$data_type`: The type `T` of the data in the matrix in the test.
/// * `$data_self`: The actual data array for the matrix in the test, must have a length of `6`.
/// * `$data_other`: The scalar value of `other`.
/// * `$operator`: The operator of the scalar assign operation.
/// * `$expected_result`: An array of expected values for the operation in the test.
///
/// # Example
///
/// Implement tests for the addition of a `Matrix<T>` to which a `T` is added:
///
/// ```text
/// test_scalar_assign_operator!(
///     owned,
///     f64,
///     [0.0, 2.3, -1.2, 42.1337, 1.0, -4.4],
///     0.1,
///     +=,
///     [0.1, 2.4, -1.1, 42.2337, 1.1, -4.3]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! test_scalar_assign_operator {
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

            /// Test the assign operator.
            #[test]
            fn correct_dimensions() {
                let rows: NonZeroUsize = NonZeroUsize::new(2).unwrap();
                let columns: NonZeroUsize = NonZeroUsize::new(3).unwrap();
                let data_self: [$data_type; 6] = $data_self;
                let other: $data_type = $data_other;
                let mut matrix = Matrix::from_slice(rows, columns, &data_self).unwrap();

                matrix $operator other;
                assert_eq!(matrix.as_slice(), $expected_result);
            }
        }
    };
}

// endregion

// region Documentation

/// Get a documentation string for the scalar assign operators.
///
/// # Parameters
///
/// * `$explanation`: A short explanation of what the operator does.
/// * `$data_type`: The type `T` of the data in the matrix in the example.
/// * `$data_self`: The actual data array for the matrix in the example. It must have a length of
///                 `6`.
/// * `$data_other`: The scalar value added to the matrix in the example.
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
///     +=,
///     [1.4, -1.03, 2.3, 4.6, 1.3, 43.4337]
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! doc_scalar_assign_operator {
    ($explanation:expr,
     $data_type:tt,
     $data_self:expr,
     $data_other:expr,
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
            "let mut matrix = Matrix::from_slice(rows, columns, &data_matrix).unwrap();",
            "\n\n",
            "matrix ",
            stringify!($operator),
            " other;\n",
            "assert_eq!(matrix.as_slice(), &",
            stringify!($expected_result),
            ");\n",
            "```"
        );
    };
}

// endregion
