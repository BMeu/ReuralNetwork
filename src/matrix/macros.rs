// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! General macros for matrices.

/// Specify the `Matrix<T>` type either as owned or as referenced.
///
/// # Parameters
///
/// * `$token`: Either `*` for owned types or `&` for referenced types.
///
/// # Example
///
/// Implement `Add<Matrix<T>> for &'_ Matrix<T>` with output type `Matrix<T>`:
///
/// ```
/// use std::ops::Add;
/// # use reural_network::specify_matrix_type;
/// #
/// # struct Matrix<T> {
/// #     data: Vec<T>,
/// # };
///
/// impl<T> Add<specify_matrix_type!(*)> for specify_matrix_type!(&) {
///     type Output = specify_matrix_type!(*);
///
///     fn add(self, other: specify_matrix_type!(*)) -> Self::Output {
///         return other;
///     }
/// }
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! specify_matrix_type {
    ($token:tt) => {
        $crate::specify_type!($token Matrix<T>)
    }
}
