pub fn format_decimal(value: f64) -> String {
    let mut formatted = format!("{value:.2}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.pop();
    }
    formatted
}

pub fn format_optional_decimal(value: Option<f64>) -> String {
    value
        .map(format_decimal)
        .unwrap_or_else(|| "unknown".to_string())
}

pub fn format_u64(value: u64) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().rev().enumerate() {
        if index != 0 && index % 3 == 0 {
            formatted.push(',');
        }
        formatted.push(ch);
    }
    formatted.chars().rev().collect()
}

pub fn format_file_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    let bytes = bytes as f64;
    if bytes >= GB {
        format!("{} GB", format_decimal(bytes / GB))
    } else if bytes >= MB {
        format!("{} MB", format_decimal(bytes / MB))
    } else {
        format!("{} KB", format_decimal(bytes / KB))
    }
}
