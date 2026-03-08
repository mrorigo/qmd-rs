// Rust guideline compliant 2026-03-08

/// Configurable markdown chunking parameters.
#[derive(Debug, Clone, Copy)]
pub struct ChunkerConfig {
    /// Approximate target tokens per chunk.
    pub target_tokens: usize,
    /// Fractional overlap between adjacent chunks.
    pub overlap_ratio: f32,
    /// Lookback window (in tokens) used for boundary scoring.
    pub lookback_tokens: usize,
    /// Max overshoot while preserving code-fence integrity.
    pub code_overshoot_tokens: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            target_tokens: 900,
            overlap_ratio: 0.15,
            lookback_tokens: 200,
            code_overshoot_tokens: 400,
        }
    }
}

/// One markdown chunk with source line range and token estimate.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Chunk text.
    pub content: String,
    /// Approximate token count.
    pub token_count: usize,
    /// One-based start line.
    pub start_line: usize,
    /// One-based end line.
    pub end_line: usize,
}

#[derive(Debug, Clone)]
struct LineUnit {
    text: String,
    tokens: usize,
    cumulative_tokens_end: usize,
    breakpoint_score: f32,
    in_code_block_end: bool,
}

/// Split markdown into semantically weighted chunks.
///
/// # Arguments
/// `markdown` - Raw markdown input.
/// `cfg` - Chunking parameters.
///
/// # Returns
/// Chunk list preserving markdown boundaries where possible.
pub fn chunk_markdown(markdown: &str, cfg: ChunkerConfig) -> Vec<Chunk> {
    let lines: Vec<&str> = markdown.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut units = Vec::with_capacity(lines.len());
    let mut cumulative = 0usize;
    let mut in_code = false;

    for line in &lines {
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
        }

        let tokens = estimate_tokens(line);
        cumulative = cumulative.saturating_add(tokens);

        units.push(LineUnit {
            text: (*line).to_string(),
            tokens,
            cumulative_tokens_end: cumulative,
            breakpoint_score: score_breakpoint(line),
            in_code_block_end: in_code,
        });
    }

    let total_tokens = units.last().map(|u| u.cumulative_tokens_end).unwrap_or(0);
    let mut chunks = Vec::new();
    let mut start_idx = 0usize;

    while start_idx < units.len() {
        let start_token = if start_idx == 0 {
            0
        } else {
            units[start_idx - 1].cumulative_tokens_end
        };

        let target_abs = start_token.saturating_add(cfg.target_tokens);
        let mut end_idx = find_first_reaching_target(&units, start_idx, target_abs);

        if end_idx + 1 >= units.len() {
            end_idx = units.len() - 1;
        } else {
            let mut end_token = units[end_idx].cumulative_tokens_end;
            if units[end_idx].in_code_block_end {
                let max_allowed = target_abs.saturating_add(cfg.code_overshoot_tokens);
                while end_idx + 1 < units.len()
                    && units[end_idx].in_code_block_end
                    && end_token < max_allowed
                {
                    end_idx += 1;
                    end_token = units[end_idx].cumulative_tokens_end;
                }
            }

            let window_start_token = end_token.saturating_sub(cfg.lookback_tokens);
            let best_break =
                best_breakpoint(&units, start_idx, end_idx, end_token, window_start_token);
            end_idx = best_break.unwrap_or(end_idx);
        }

        let chunk = build_chunk(&units, start_idx, end_idx);
        if !chunk.content.trim().is_empty() {
            chunks.push(chunk);
        }

        if end_idx + 1 >= units.len() {
            break;
        }

        let overlap_tokens = (cfg.target_tokens as f32 * cfg.overlap_ratio) as usize;
        if overlap_tokens == 0 {
            start_idx = end_idx + 1;
            continue;
        }

        let end_abs = units[end_idx].cumulative_tokens_end;
        let desired_start_token = end_abs.saturating_sub(overlap_tokens);
        start_idx = find_start_idx_for_token(&units, start_idx + 1, desired_start_token);

        if start_idx >= units.len() {
            break;
        }

        if units[start_idx].cumulative_tokens_end >= total_tokens {
            break;
        }
    }

    chunks
}

fn build_chunk(units: &[LineUnit], start_idx: usize, end_idx: usize) -> Chunk {
    let mut content = String::new();
    let mut tokens = 0usize;

    for unit in units.iter().take(end_idx + 1).skip(start_idx) {
        content.push_str(&unit.text);
        content.push('\n');
        tokens = tokens.saturating_add(unit.tokens);
    }

    Chunk {
        content,
        token_count: tokens,
        start_line: start_idx + 1,
        end_line: end_idx + 1,
    }
}

fn find_first_reaching_target(units: &[LineUnit], start_idx: usize, target_abs: usize) -> usize {
    for (idx, unit) in units.iter().enumerate().skip(start_idx) {
        if unit.cumulative_tokens_end >= target_abs {
            return idx;
        }
    }
    units.len().saturating_sub(1)
}

fn best_breakpoint(
    units: &[LineUnit],
    start_idx: usize,
    end_idx: usize,
    current_token_abs: usize,
    window_start_token: usize,
) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (idx, unit) in units.iter().enumerate().take(end_idx + 1).skip(start_idx) {
        let bp_token = unit.cumulative_tokens_end;
        if bp_token < window_start_token {
            continue;
        }

        let distance = current_token_abs.saturating_sub(bp_token) as f32;
        let window = (current_token_abs.saturating_sub(window_start_token)).max(1) as f32;
        let decay = 1.0 - ((distance / window).powi(2) * 0.7);
        let score = unit.breakpoint_score * decay;

        match best {
            Some((_, top)) if score <= top => {}
            _ => best = Some((idx, score)),
        }
    }

    best.map(|(idx, _)| idx)
}

fn find_start_idx_for_token(units: &[LineUnit], min_idx: usize, token_abs: usize) -> usize {
    for (idx, unit) in units.iter().enumerate().skip(min_idx) {
        if unit.cumulative_tokens_end >= token_abs {
            return idx;
        }
    }
    units.len().saturating_sub(1)
}

fn score_breakpoint(line: &str) -> f32 {
    let trimmed = line.trim_start();
    if trimmed.starts_with("# ") {
        return 100.0;
    }
    if trimmed.starts_with("## ") {
        return 90.0;
    }
    if trimmed.starts_with("```") {
        return 80.0;
    }
    if trimmed.is_empty() {
        return 20.0;
    }
    1.0
}

fn estimate_tokens(line: &str) -> usize {
    let words = line.split_whitespace().count();
    words.max(1)
}
