//! Built-in telemetry. ulview-ready.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanStatus {
    Ok,
    Error,
    Timeout,
}

impl std::fmt::Display for SpanStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ok => write!(f, "ok"),
            Self::Error => write!(f, "error"),
            Self::Timeout => write!(f, "timeout"),
        }
    }
}

pub struct Span {
    pub name: String,
    pub run_id: String,
    pub step_name: Option<String>,
    start: Instant,
    pub attributes: HashMap<String, String>,
    pub events: Vec<(u64, String)>,
}

impl Span {
    pub fn new(name: impl Into<String>, run_id: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            run_id: run_id.into(),
            step_name: None,
            start: Instant::now(),
            attributes: HashMap::new(),
            events: Vec::new(),
        }
    }
    pub fn for_step(mut self, step: &str) -> Self {
        self.step_name = Some(step.to_string());
        self
    }
    pub fn attr(mut self, k: &str, v: &str) -> Self {
        self.attributes.insert(k.into(), v.into());
        self
    }
    pub fn add_event(&mut self, msg: impl Into<String>) {
        self.events
            .push((self.start.elapsed().as_millis() as u64, msg.into()));
    }
    pub fn finish(self, status: SpanStatus) -> SpanRecord {
        SpanRecord {
            name: self.name,
            run_id: self.run_id,
            step_name: self.step_name,
            status,
            latency_ms: self.start.elapsed().as_millis() as u64,
            attributes: self.attributes,
            events: self.events,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpanRecord {
    pub name: String,
    pub run_id: String,
    pub step_name: Option<String>,
    pub status: SpanStatus,
    pub latency_ms: u64,
    pub attributes: HashMap<String, String>,
    pub events: Vec<(u64, String)>,
}

#[derive(Debug, Clone)]
pub struct TelemetrySummary {
    pub total_spans: usize,
    pub error_spans: usize,
    pub avg_latency_ms: u64,
}

pub struct Telemetry {
    spans: Mutex<Vec<SpanRecord>>,
}

impl Telemetry {
    pub fn new() -> Self {
        Self {
            spans: Mutex::new(Vec::new()),
        }
    }
    pub fn record(&self, span: SpanRecord) {
        self.spans.lock().unwrap().push(span);
    }
    pub fn spans(&self) -> Vec<SpanRecord> {
        self.spans.lock().unwrap().clone()
    }
    pub fn summary(&self) -> TelemetrySummary {
        let spans = self.spans.lock().unwrap();
        let total = spans.len();
        let errors = spans
            .iter()
            .filter(|s| s.status == SpanStatus::Error)
            .count();
        let avg = if total > 0 {
            spans.iter().map(|s| s.latency_ms).sum::<u64>() / total as u64
        } else {
            0
        };
        TelemetrySummary {
            total_spans: total,
            error_spans: errors,
            avg_latency_ms: avg,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_status_display() {
        assert_eq!(SpanStatus::Ok.to_string(), "ok");
        assert_eq!(SpanStatus::Error.to_string(), "error");
    }

    #[test]
    fn telemetry_collect() {
        let tel = Telemetry::new();
        for i in 0..5 {
            tel.record(Span::new(format!("op_{}", i), "r1").finish(if i < 4 {
                SpanStatus::Ok
            } else {
                SpanStatus::Error
            }));
        }
        let s = tel.summary();
        assert_eq!(s.total_spans, 5);
        assert_eq!(s.error_spans, 1);
    }
}
