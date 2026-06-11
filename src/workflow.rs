//! Workflow definition: the complete DAG of steps.

use crate::context::ContextValue;
use crate::error::FlowError;
use crate::step::Step;
use std::collections::{HashMap, HashSet};

/// Workflow execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowMode {
    /// Linear pipeline: steps run in declaration order.
    Pipeline,
    /// DAG: steps run based on dependency graph (parallel where possible).
    Graph,
    /// State machine: steps run based on state transitions.
    StateMachine,
}

/// Overall workflow status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowStatus {
    Created,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Paused,
}

impl std::fmt::Display for FlowStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Created => write!(f, "created"),
            Self::Running => write!(f, "running"),
            Self::Succeeded => write!(f, "succeeded"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Paused => write!(f, "paused"),
        }
    }
}

/// Input to a workflow run.
#[derive(Debug, Clone, Default)]
pub struct FlowInput {
    pub vars: HashMap<String, ContextValue>,
}

impl FlowInput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn var(mut self, key: impl Into<String>, value: impl Into<ContextValue>) -> Self {
        self.vars.insert(key.into(), value.into());
        self
    }
}

/// Output from a workflow run.
#[derive(Debug, Clone)]
pub struct FlowOutput {
    pub run_id: String,
    pub status: FlowStatus,
    pub outputs: HashMap<String, ContextValue>,
    pub steps_completed: usize,
    pub steps_failed: usize,
    pub steps_skipped: usize,
    pub tokens_used: usize,
    pub latency_ms: u64,
    pub error: Option<FlowError>,
}

impl FlowOutput {
    pub fn get(&self, path: &str) -> Option<&ContextValue> {
        self.outputs.get(path)
    }

    pub fn get_str(&self, path: &str) -> Option<&str> {
        self.get(path)?.as_str()
    }

    pub fn succeeded(&self) -> bool {
        self.status == FlowStatus::Succeeded
    }
}

/// A complete workflow definition.
#[derive(Debug, Clone)]
pub struct Flow {
    pub name: String,
    pub description: String,
    pub mode: FlowMode,
    pub steps: Vec<Step>,
    pub context_budget: usize,
    pub persistent: bool,
    pub max_retries: usize,
    pub timeout_ms: Option<u64>,
}

impl Flow {
    pub fn pipeline(name: impl Into<String>) -> FlowBuilder {
        FlowBuilder::new(name, FlowMode::Pipeline)
    }

    pub fn graph(name: impl Into<String>) -> FlowBuilder {
        FlowBuilder::new(name, FlowMode::Graph)
    }

    pub fn state_machine(name: impl Into<String>) -> FlowBuilder {
        FlowBuilder::new(name, FlowMode::StateMachine)
    }

    /// Validate the workflow definition.
    pub fn validate(&self) -> Result<(), FlowError> {
        if self.steps.is_empty() {
            return Err(FlowError::InvalidFlow("workflow has no steps".into()));
        }

        let step_names: HashSet<&str> = self.steps.iter().map(|s| s.name.as_str()).collect();

        // Check for duplicate step names
        if step_names.len() != self.steps.len() {
            return Err(FlowError::InvalidFlow("duplicate step names".into()));
        }

        // Check dependencies exist
        for step in &self.steps {
            for dep in &step.depends_on {
                if !step_names.contains(dep.as_str()) {
                    return Err(FlowError::InvalidFlow(format!(
                        "step {:?} depends on unknown step {:?}",
                        step.name, dep
                    )));
                }
            }
        }

        // Check for cycles (DFS)
        if self.mode == FlowMode::Graph {
            let dep_map: HashMap<&str, Vec<&str>> = self
                .steps
                .iter()
                .map(|s| {
                    (
                        s.name.as_str(),
                        s.depends_on.iter().map(|d| d.as_str()).collect(),
                    )
                })
                .collect();

            let mut visited: HashSet<&str> = HashSet::new();
            let mut in_stack: HashSet<&str> = HashSet::new();

            fn has_cycle<'a>(
                node: &'a str,
                dep_map: &HashMap<&'a str, Vec<&'a str>>,
                visited: &mut HashSet<&'a str>,
                in_stack: &mut HashSet<&'a str>,
            ) -> bool {
                visited.insert(node);
                in_stack.insert(node);
                if let Some(deps) = dep_map.get(node) {
                    for &dep in deps {
                        if !visited.contains(dep) && has_cycle(dep, dep_map, visited, in_stack) {
                            return true;
                        }
                        if in_stack.contains(dep) {
                            return true;
                        }
                    }
                }
                in_stack.remove(node);
                false
            }

            for step in &self.steps {
                if !visited.contains(step.name.as_str())
                    && has_cycle(&step.name, &dep_map, &mut visited, &mut in_stack)
                {
                    return Err(FlowError::InvalidFlow(format!(
                        "cycle detected involving step {:?}",
                        step.name
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get steps that are ready to run (all dependencies completed).
    pub fn ready_steps<'a>(&'a self, completed: &HashSet<&str>) -> Vec<&'a Step> {
        self.steps
            .iter()
            .filter(|s| {
                !completed.contains(s.name.as_str())
                    && s.depends_on.iter().all(|d| completed.contains(d.as_str()))
            })
            .collect()
    }
}

/// Builder for workflows.
pub struct FlowBuilder {
    name: String,
    description: String,
    mode: FlowMode,
    steps: Vec<Step>,
    context_budget: usize,
    persistent: bool,
    max_retries: usize,
    timeout_ms: Option<u64>,
}

impl FlowBuilder {
    pub fn new(name: impl Into<String>, mode: FlowMode) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            mode,
            steps: Vec::new(),
            context_budget: 8192,
            persistent: false,
            max_retries: 0,
            timeout_ms: None,
        }
    }

    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }

    pub fn step(mut self, step: Step) -> Self {
        self.steps.push(step);
        self
    }

    pub fn context_budget(mut self, tokens: usize) -> Self {
        self.context_budget = tokens;
        self
    }

    pub fn persistent(mut self, p: bool) -> Self {
        self.persistent = p;
        self
    }

    pub fn max_retries(mut self, n: usize) -> Self {
        self.max_retries = n;
        self
    }

    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    pub fn build(self) -> Result<Flow, FlowError> {
        let flow = Flow {
            name: self.name,
            description: self.description,
            mode: self.mode,
            steps: self.steps,
            context_budget: self.context_budget,
            persistent: self.persistent,
            max_retries: self.max_retries,
            timeout_ms: self.timeout_ms,
        };
        flow.validate()?;
        Ok(flow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::{Input, Step};

    fn simple_flow() -> Flow {
        Flow::pipeline("test")
            .step(
                Step::tool("s1")
                    .tool("echo")
                    .input_literal("text", "hello")
                    .build(),
            )
            .step(
                Step::tool("s2")
                    .tool("echo")
                    .input("text", Input::from_step("s1.output"))
                    .depends_on("s1")
                    .build(),
            )
            .build()
            .unwrap()
    }

    #[test]
    fn flow_builds() {
        let flow = simple_flow();
        assert_eq!(flow.name, "test");
        assert_eq!(flow.steps.len(), 2);
    }

    #[test]
    fn flow_validate_empty() {
        let result = Flow::pipeline("empty").build();
        assert!(result.is_err());
    }

    #[test]
    fn flow_validate_duplicate_steps() {
        let result = Flow::pipeline("dup")
            .step(
                Step::tool("s1")
                    .tool("echo")
                    .input_literal("text", "a")
                    .build(),
            )
            .step(
                Step::tool("s1")
                    .tool("echo")
                    .input_literal("text", "b")
                    .build(),
            )
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn flow_validate_unknown_dep() {
        let result = Flow::graph("bad")
            .step(
                Step::tool("s1")
                    .tool("echo")
                    .input_literal("text", "a")
                    .depends_on("nonexistent")
                    .build(),
            )
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn flow_ready_steps() {
        let flow = simple_flow();
        let mut completed = HashSet::new();

        // Initially only s1 is ready
        let ready = flow.ready_steps(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].name, "s1");

        // After s1 completes, s2 is ready
        completed.insert("s1");
        let ready2 = flow.ready_steps(&completed);
        assert_eq!(ready2.len(), 1);
        assert_eq!(ready2[0].name, "s2");
    }

    #[test]
    fn flow_input() {
        let input = FlowInput::new()
            .var("task", "refactor auth")
            .var("limit", 10i64);
        assert_eq!(input.vars.len(), 2);
    }

    #[test]
    fn flow_status_display() {
        assert_eq!(FlowStatus::Succeeded.to_string(), "succeeded");
        assert_eq!(FlowStatus::Failed.to_string(), "failed");
    }
}
