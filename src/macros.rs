// Copyright 2019 Bastian Meyer
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or
// distributed except according to those terms.

//! Collection of general macros.

/// Specify a type either as owned or as referenced.
///
/// # Parameters
///
/// * `$type`: The type to specify either as owned or as referenced.
///
/// # Example
///
/// Implement `Add<Example> for &'_ Example` with output type `Example`:
///
/// ```
/// use std::ops::Add;
/// # use reural_network::specify_type;
///
/// struct Example;
///
/// impl Add<specify_type!(* Example)> for specify_type!(& Example) {
///     type Output = specify_type!(* Example);
///
///     fn add(self, other: specify_type!(* Example)) -> Self::Output {
///         return other;
///     }
/// }
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! specify_type {
    // Specify the type as an owned type.
    (* $type:ty) => {
        $type
    };

    // Specify the type as a referenced type.
    (& $type:ty) => {
        &'_ $type
    };
}

/// Access the given variable either by value or by reference.
///
/// # Parameters
///
/// * `$variable`: The variable to get either by value or as a reference.
///
/// # Example
///
/// ```
/// # use reural_network::access_variable;
/// # fn main() {
/// let a = [0, 2, 4, 8];
///
/// // Access `a` by value.
/// let b = access_variable!(* a);
/// // Is the same as:
/// // let b = a;
///
/// // Access `a` by reference.
/// let c = access_variable!(& a);
/// // Is the same as:
/// // let c = &a;
/// # }
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! access_variable {
    // Get the variable by value.
    (* $variable:ident) => {
        $variable
    };

    // Get the variable by reference.
    (& $variable:ident) => {
        &$variable
    };
}
