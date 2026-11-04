//! Protected CLI and integration pipeline. Port the eight sibling modules.
mod money;
mod date;
mod csv;
mod labels;
mod tags;
mod filter;
mod summary;
mod audit;

#[derive(Clone, Debug)]
pub struct Record {
    pub id: String,
    pub date: String,
    pub vendor: String,
    pub cents: i64,
    pub tags: Vec<String>,
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"), '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"'); out
}
fn json_list(v: &[String]) -> String { format!("[{}]", v.iter().map(|s| json_string(s)).collect::<Vec<_>>().join(",")) }
fn totals_json(v: &[(String, usize, i64)]) -> String {
    format!("[{}]", v.iter().map(|(name,count,cents)| format!("{{\"vendor\":{},\"count\":{},\"total\":{}}}", json_string(name), count, json_string(&money::format(*cents)))).collect::<Vec<_>>().join(","))
}
fn load(path: &str) -> Result<(Vec<Record>,usize),String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut rows = Vec::new(); let mut invalid = 0;
    for line in text.lines().skip(1) {
        let row = (|| {
            let f = csv::parse_line(line)?; if f.len() != 5 { return None; }
            let id = f[0].trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{000b}' | '\u{000c}')).to_owned();
            let vendor = labels::normalize(&f[2]);
            if id.is_empty() || vendor.is_empty() || !date::valid(&f[1]) { return None; }
            Some(Record { id, date:f[1].clone(), vendor, cents:money::parse(&f[3])?, tags:tags::parse(&f[4])? })
        })();
        if let Some(row) = row { rows.push(row); } else { invalid += 1; }
    }
    Ok((rows, invalid))
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let result: Result<String,String> = (|| {
        let command = a.get(1).ok_or("missing command")?.as_str();
        let value = a.get(2).ok_or("missing argument")?;
        match command {
            "money" => Ok(money::parse(value).map(money::format).unwrap_or("ERR".into())),
            "date" => Ok(if date::valid(value) { value.clone() } else { "ERR".into() }),
            "csv" => Ok(csv::parse_line(value).map(|v| json_list(&v)).unwrap_or("ERR".into())),
            "label" => Ok(labels::normalize(value)),
            "tags" => Ok(tags::parse(value).map(|v| json_list(&v)).unwrap_or("ERR".into())),
            "filter" => {
                let to = a.get(3).ok_or("missing end")?; let path = a.get(4).ok_or("missing path")?;
                if !date::valid(value) || !date::valid(to) || value > to { return Ok("ERR".into()); }
                let (rows,_) = load(path)?; let (rows,_) = audit::dedupe(&rows);
                Ok(json_list(&filter::between(&rows,value,to).iter().map(|r|r.id.clone()).collect::<Vec<_>>()))
            },
            "summary" => { let (rows,_) = load(value)?; let (rows,_) = audit::dedupe(&rows); Ok(totals_json(&summary::group(&rows))) },
            "audit" => { let (rows,_) = load(value)?; let (_,ids) = audit::dedupe(&rows); Ok(json_list(&ids)) },
            "report" => {
                let from = a.get(3).ok_or("missing start")?; let to = a.get(4).ok_or("missing end")?;
                if !date::valid(from) || !date::valid(to) || from > to { return Ok("ERR".into()); }
                let (rows,invalid) = load(value)?; let (rows,duplicates) = audit::dedupe(&rows);
                let selected = filter::between(&rows,from,to);
                Ok(format!("{{\"accepted\":{},\"invalid\":{},\"duplicates\":{},\"selected\":{},\"totals\":{}}}", rows.len(),invalid,json_list(&duplicates),json_list(&selected.iter().map(|r|r.id.clone()).collect::<Vec<_>>()),totals_json(&summary::group(&selected))))
            },
            _ => Err("unknown command".into()),
        }
    })();
    match result { Ok(s) => println!("{}",s), Err(e) => { eprintln!("{}",e); std::process::exit(2); } }
}
