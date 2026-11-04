//! TODO: Port reference.c dedupe_rows. Keep FIRST record for each exact ID in
//! original order. Return each repeated ID once, sorted lexicographically, in
//! the second vector. IDs are case sensitive. Do not mutate the input.
use crate::Record;
pub fn dedupe(_rows: &[Record]) -> (Vec<Record>, Vec<String>) { (Vec::new(),Vec::new()) }
