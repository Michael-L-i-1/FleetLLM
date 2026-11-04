//! TODO: Port reference.c money functions. Do not use floating point.
//! parse accepts optional +/- then >=1 ASCII digit, optionally '.' with 1 or 2
//! digits. Reject whitespace, exponents, missing parts, extra precision, overflow.
//! Return exact cents (i64 range). format always emits two decimals, no '+' or
//! redundant leading zeroes; formatting i64::MIN must also work.
pub fn parse(_text: &str) -> Option<i64> { None }
pub fn format(_cents: i64) -> String { String::new() }
