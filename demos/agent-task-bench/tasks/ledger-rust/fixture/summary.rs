//! TODO: Port reference.c group_rows. Group by exact vendor string, count rows,
//! sum signed cents, and sort by vendor lexicographically. Include zero totals.
//! Inputs in this task never overflow i64 when summed.
use crate::Record;
pub fn group(_rows: &[Record]) -> Vec<(String, usize, i64)> { Vec::new() }
