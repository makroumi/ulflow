//! Event system for workflow observability.

use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventKind {
    FlowStarted,
    FlowSucceeded,
    FlowFailed,
    FlowCancelled,
    StepStarted,
    StepSucceeded,
    StepFailed,
    StepSkipped,
    StepRetrying,
    BudgetWarning,
    CheckpointSaved,
    CheckpointLoaded,
    AgentCalled,
    ToolCalled,
}

impl std::fmt::Display for EventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::FlowStarted => "flow.started",
            Self::FlowSucceeded => "flow.succeeded",
            Self::FlowFailed => "flow.failed",
            Self::FlowCancelled => "flow.cancelled",
            Self::StepStarted => "step.started",
            Self::StepSucceeded => "step.succeeded",
            Self::StepFailed => "step.failed",
            Self::StepSkipped => "step.skipped",
            Self::StepRetrying => "step.retrying",
            Self::BudgetWarning => "budget.warning",
            Self::CheckpointSaved => "checkpoint.saved",
            Self::CheckpointLoaded => "checkpoint.loaded",
            Self::AgentCalled => "agent.called",
            Self::ToolCalled => "tool.called",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone)]
pub struct FlowEvent {
    pub kind: EventKind,
    pub run_id: String,
    pub step_name: Option<String>,
    pub message: String,
    pub timestamp_ms: u64,
    pub metadata: std::collections::HashMap<String, String>,
}

impl FlowEvent {
    pub fn new(kind: EventKind, run_id: &str) -> Self {
        Self {
            kind,
            run_id: run_id.to_string(),
            step_name: None,
            message: String::new(),
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_step(mut self, step: &str) -> Self {
        self.step_name = Some(step.to_string());
        self
    }
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = msg.into();
        self
    }
    pub fn with_meta(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

pub type EventHandler = Box<dyn Fn(&FlowEvent) + Send + Sync>;

pub struct EventBus {
    handlers: Vec<EventHandler>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }
    pub fn subscribe(&mut self, handler: EventHandler) {
        self.handlers.push(handler);
    }
    pub fn emit(&self, event: &FlowEvent) {
        for h in &self.handlers {
            h(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn event_kind_display() {
        assert_eq!(EventKind::FlowStarted.to_string(), "flow.started");
        assert_eq!(EventKind::StepFailed.to_string(), "step.failed");
    }

    #[test]
    fn event_creation() {
        let e = FlowEvent::new(EventKind::StepStarted, "run_1")
            .with_step("search")
            .with_meta("tool", "code_search");
        assert_eq!(e.run_id, "run_1");
        assert_eq!(e.step_name.as_deref(), Some("search"));
        assert!(e.timestamp_ms > 0);
    }

    #[test]
    fn event_bus() {
        let received = Arc::new(Mutex::new(Vec::<String>::new()));
        let rc = Arc::clone(&received);
        let mut bus = EventBus::new();
        bus.subscribe(Box::new(move |e| {
            rc.lock().unwrap().push(e.kind.to_string())
        }));
        bus.emit(&FlowEvent::new(EventKind::FlowStarted, "r1"));
        bus.emit(&FlowEvent::new(EventKind::FlowSucceeded, "r1"));
        let events = received.lock().unwrap();
        assert_eq!(events.len(), 2);
    }
}
