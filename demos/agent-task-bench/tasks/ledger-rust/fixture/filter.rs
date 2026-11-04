//! TODO: Port reference.c filter_rows. Return cloned records with inclusive
//! from <= date <= to, preserving input order. Invalid endpoints or from>to
//! return an empty vector. Use crate::date::valid for endpoint checks.
use crate::Record;
pub fn between(_rows: &[Record], _from: &str, _to: &str) -> Vec<Record> { Vec::new() }
