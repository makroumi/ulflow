//! Workflow checkpointing.

use crate::context::ContextValue;
use crate::step::StepStatus;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub run_id: String,
    pub flow_name: String,
    pub created_at_ms: u64,
    pub completed_steps: Vec<String>,
    pub step_statuses: HashMap<String, StepStatus>,
    pub context_outputs: HashMap<String, String>,
    pub tokens_used: usize,
}

impl Checkpoint {
    pub fn new(run_id: &str, flow_name: &str) -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            run_id: run_id.to_string(),
            flow_name: flow_name.to_string(),
            created_at_ms: now_ms,
            completed_steps: Vec::new(),
            step_statuses: HashMap::new(),
            context_outputs: HashMap::new(),
            tokens_used: 0,
        }
    }

    pub fn mark_completed(&mut self, step: &str, status: StepStatus) {
        self.completed_steps.push(step.to_string());
        self.step_statuses.insert(step.to_string(), status);
    }

    pub fn is_step_completed(&self, step: &str) -> bool {
        self.completed_steps.contains(&step.to_string())
    }

    pub fn save_context_value(&mut self, key: &str, value: &ContextValue) {
        let s = match value {
            ContextValue::String(s) => s.clone(),
            ContextValue::Integer(i) => i.to_string(),
            ContextValue::Boolean(b) => b.to_string(),
            _ => format!("{:?}", value),
        };
        self.context_outputs.insert(key.to_string(), s);
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut parts = vec![
            format!("run_id:{}", self.run_id),
            format!("flow:{}", self.flow_name),
            format!("ts:{}", self.created_at_ms),
            format!("tokens:{}", self.tokens_used),
            format!("completed:{}", self.completed_steps.join(",")),
        ];
        for (k, v) in &self.context_outputs {
            parts.push(format!("out:{}={}", k, v));
        }
        parts.join("\n").into_bytes()
    }

    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        let text = std::str::from_utf8(data).ok()?;
        let mut cp = Checkpoint::new("", "");
        for line in text.lines() {
            if let Some(v) = line.strip_prefix("run_id:") {
                cp.run_id = v.into();
            } else if let Some(v) = line.strip_prefix("flow:") {
                cp.flow_name = v.into();
            } else if let Some(v) = line.strip_prefix("ts:") {
                cp.created_at_ms = v.parse().unwrap_or(0);
            } else if let Some(v) = line.strip_prefix("tokens:") {
                cp.tokens_used = v.parse().unwrap_or(0);
            } else if let Some(v) = line.strip_prefix("completed:") {
                cp.completed_steps = if v.is_empty() {
                    vec![]
                } else {
                    v.split(',').map(String::from).collect()
                };
            } else if let Some(v) = line.strip_prefix("out:") {
                if let Some((k, val)) = v.split_once('=') {
                    cp.context_outputs.insert(k.into(), val.into());
                }
            }
        }
        Some(cp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut cp = Checkpoint::new("run_1", "my_flow");
        cp.mark_completed("search", StepStatus::Succeeded);
        cp.tokens_used = 500;
        cp.save_context_value("search.output", &ContextValue::String("auth.py".into()));
        let bytes = cp.to_bytes();
        let r = Checkpoint::from_bytes(&bytes).unwrap();
        assert_eq!(r.run_id, "run_1");
        assert_eq!(r.flow_name, "my_flow");
        assert_eq!(r.tokens_used, 500);
        assert!(r.is_step_completed("search"));
        assert!(!r.is_step_completed("write"));
    }
}
