//! TODO: Port reference.c csv_parse. One CSV record, comma separated; empty
//! fields allowed. Quotes only at field start, doubled quotes escape a quote,
//! quoted fields may contain commas. After closing quote only comma/end allowed.
//! Reject unclosed/stray quotes and CR/LF anywhere. Preserve all field contents.
pub fn parse_line(_text: &str) -> Option<Vec<String>> { None }
