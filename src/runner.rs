//! FlowRunner: executes workflow DAGs.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use ulmcp::registry::Registry;

use crate::context::{ContextValue, ExecutionContext};
use crate::error::{FlowError, StepError};
use crate::event::{EventBus, EventKind, FlowEvent};
use crate::step::{StepKind, StepResult, StepStatus};
use crate::telemetry::{Span, SpanStatus, Telemetry};
use crate::workflow::{Flow, FlowInput, FlowOutput, FlowStatus};

/// Executes workflow flows against a tool registry.
pub struct FlowRunner {
    registry: Registry,
    telemetry: Telemetry,
    event_bus: EventBus,
}

impl FlowRunner {
    pub fn new(registry: Registry) -> Self {
        Self {
            registry,
            telemetry: Telemetry::new(),
            event_bus: EventBus::new(),
        }
    }

    pub fn on_event<F: Fn(&FlowEvent) + Send + Sync + 'static>(&mut self, handler: F) {
        self.event_bus.subscribe(Box::new(handler));
    }

    /// Execute a flow synchronously.
    pub fn run(&mut self, flow: Flow, input: FlowInput) -> Result<FlowOutput, FlowError> {
        let run_id = format!("run_{}", chrono_ms());
        let start = Instant::now();

        // Validate
        flow.validate()?;

        // Build context
        let mut ctx = ExecutionContext::new(&run_id, flow.context_budget);
        for (k, v) in input.vars {
            ctx.set(k, v);
        }

        self.emit(
            FlowEvent::new(EventKind::FlowStarted, &run_id)
                .with_message(format!("Starting flow: {}", flow.name)),
        );

        let mut step_results: Vec<StepResult> = Vec::new();
        let mut completed: HashSet<&str> = HashSet::new();
        let mut failed = false;
        let mut flow_error: Option<FlowError> = None;

        // Execute based on mode
        'outer: loop {
            let ready = flow.ready_steps(&completed);
            if ready.is_empty() {
                break;
            }

            for step in ready {
                // Check condition
                if let Some(cond) = &step.condition {
                    let val = ctx.get(cond).cloned().unwrap_or(ContextValue::Null);
                    if !val.is_truthy() {
                        self.emit(
                            FlowEvent::new(EventKind::StepSkipped, &run_id).with_step(&step.name),
                        );
                        step_results.push(StepResult {
                            step_name: step.name.clone(),
                            status: StepStatus::Skipped,
                            output: None,
                            error: None,
                            tokens_used: 0,
                            latency_ms: 0,
                            attempts: 0,
                        });
                        completed.insert(&step.name);
                        continue;
                    }
                }

                // Execute step
                self.emit(FlowEvent::new(EventKind::StepStarted, &run_id).with_step(&step.name));

                let span = Span::new(format!("step.{}", step.name), &run_id).for_step(&step.name);

                let step_start = Instant::now();
                let result = self.execute_step(&step.kind, &mut ctx, &run_id);
                let latency_ms = step_start.elapsed().as_millis() as u64;

                match result {
                    Ok((output, tokens)) => {
                        // Store output in context
                        if let Some(ref val) = output {
                            let field = match &step.kind {
                                StepKind::Tool { output_field, .. } => output_field.clone(),
                                StepKind::Agent { output_field, .. } => output_field.clone(),
                                _ => "output".into(),
                            };
                            ctx.set_output(&step.name, &field, val.clone());
                        }

                        self.telemetry.record(span.finish(SpanStatus::Ok));
                        self.emit(
                            FlowEvent::new(EventKind::StepSucceeded, &run_id)
                                .with_step(&step.name)
                                .with_meta("latency_ms", &latency_ms.to_string()),
                        );

                        step_results.push(StepResult {
                            step_name: step.name.clone(),
                            status: StepStatus::Succeeded,
                            output,
                            error: None,
                            tokens_used: tokens,
                            latency_ms,
                            attempts: 1,
                        });
                        completed.insert(&step.name);
                    }
                    Err(e) => {
                        self.telemetry.record(span.finish(SpanStatus::Error));
                        self.emit(
                            FlowEvent::new(EventKind::StepFailed, &run_id)
                                .with_step(&step.name)
                                .with_message(e.to_string()),
                        );

                        if step.skip_on_error {
                            step_results.push(StepResult {
                                step_name: step.name.clone(),
                                status: StepStatus::Skipped,
                                output: None,
                                error: Some(e.to_string()),
                                tokens_used: 0,
                                latency_ms,
                                attempts: 1,
                            });
                            completed.insert(&step.name);
                        } else {
                            let flow_err = FlowError::StepFailed {
                                step: step.name.clone(),
                                error: StepError::ToolFailed(e.to_string()),
                            };
                            flow_error = Some(flow_err);
                            failed = true;
                            break 'outer;
                        }
                    }
                }
            }

            // Check if all steps are done
            if completed.len() == flow.steps.len() {
                break;
            }
        }

        let total_ms = start.elapsed().as_millis() as u64;
        let status = if failed {
            FlowStatus::Failed
        } else {
            FlowStatus::Succeeded
        };

        self.emit(
            if failed {
                FlowEvent::new(EventKind::FlowFailed, &run_id)
            } else {
                FlowEvent::new(EventKind::FlowSucceeded, &run_id)
            }
            .with_meta("latency_ms", &total_ms.to_string()),
        );

        let steps_failed = step_results
            .iter()
            .filter(|s| s.status == StepStatus::Failed)
            .count();
        let steps_skipped = step_results
            .iter()
            .filter(|s| s.status == StepStatus::Skipped)
            .count();
        let steps_completed = step_results
            .iter()
            .filter(|s| s.status == StepStatus::Succeeded)
            .count();
        let total_tokens: usize = step_results.iter().map(|s| s.tokens_used).sum();

        // Collect outputs
        let outputs: HashMap<String, ContextValue> = step_results
            .iter()
            .filter_map(|s| {
                s.output
                    .clone()
                    .map(|v| (format!("{}.output", s.step_name), v))
            })
            .collect();

        if failed {
            Err(flow_error.unwrap_or(FlowError::InvalidFlow("unknown failure".into())))
        } else {
            Ok(FlowOutput {
                run_id,
                status,
                outputs,
                steps_completed,
                steps_failed,
                steps_skipped,
                tokens_used: total_tokens + ctx.budget.used,
                latency_ms: total_ms,
                error: None,
            })
        }
    }

    fn execute_step(
        &self,
        kind: &StepKind,
        ctx: &mut ExecutionContext,
        _run_id: &str,
    ) -> Result<(Option<ContextValue>, usize), String> {
        match kind {
            StepKind::Tool {
                tool_name,
                inputs,
                output_field: _,
            } => {
                // Resolve inputs
                let mut args = ulmcp::tool::ToolCall {
                    call_id: format!("call_{}", chrono_ms()),
                    tool_name: tool_name.clone(),
                    arguments: std::collections::HashMap::new(),
                };
                for (key, input) in inputs {
                    if let Some(val) = input.resolve(ctx) {
                        args.arguments
                            .insert(key.clone(), context_value_to_tool_value(val));
                    }
                }

                let result = self.registry.invoke(&args);
                let tokens = result.tokens_used.unwrap_or(0);

                match result.status {
                    ulmcp::tool::ToolStatus::Success => {
                        let output = tool_value_to_context_value(result.output);
                        Ok((Some(output), tokens))
                    }
                    _ => Err(result.error.unwrap_or_else(|| "tool failed".into())),
                }
            }
            StepKind::Agent {
                prompt,
                context_inputs: _,
                output_field: _,
            } => {
                // Render prompt with context
                let rendered = ctx.render(prompt);
                // In real usage, this calls the LLM API.
                // Here we return the rendered prompt as output (LLM integration is pluggable).
                Ok((Some(ContextValue::String(rendered)), 0))
            }
            StepKind::Condition { test, .. } => {
                let val = ctx.get(test).cloned().unwrap_or(ContextValue::Null);
                Ok((Some(ContextValue::Boolean(val.is_truthy())), 0))
            }
            StepKind::Transform {
                input,
                operation,
                output_field: _,
            } => {
                let val = ctx.get(input).cloned();
                let result = apply_transform(val, operation);
                Ok((result, 0))
            }
            StepKind::Parallel { .. } | StepKind::Wait { .. } => Ok((None, 0)),
        }
    }

    fn emit(&self, event: FlowEvent) {
        self.event_bus.emit(&event);
    }

    pub fn telemetry_summary(&self) -> crate::telemetry::TelemetrySummary {
        self.telemetry.summary()
    }
}

fn context_value_to_tool_value(v: ContextValue) -> ulmcp::tool::ToolValue {
    match v {
        ContextValue::Null => ulmcp::tool::ToolValue::Null,
        ContextValue::String(s) => ulmcp::tool::ToolValue::String(s),
        ContextValue::Integer(i) => ulmcp::tool::ToolValue::Integer(i),
        ContextValue::Float(f) => ulmcp::tool::ToolValue::Float(f),
        ContextValue::Boolean(b) => ulmcp::tool::ToolValue::Boolean(b),
        ContextValue::List(l) => {
            ulmcp::tool::ToolValue::Array(l.into_iter().map(context_value_to_tool_value).collect())
        }
        ContextValue::Bytes(b) => ulmcp::tool::ToolValue::Bytes(b),
        ContextValue::Map(m) => ulmcp::tool::ToolValue::Object(
            m.into_iter()
                .map(|(k, v)| (k, context_value_to_tool_value(v)))
                .collect(),
        ),
    }
}

fn tool_value_to_context_value(v: ulmcp::tool::ToolValue) -> ContextValue {
    match v {
        ulmcp::tool::ToolValue::Null => ContextValue::Null,
        ulmcp::tool::ToolValue::String(s) => ContextValue::String(s),
        ulmcp::tool::ToolValue::Integer(i) => ContextValue::Integer(i),
        ulmcp::tool::ToolValue::Float(f) => ContextValue::Float(f),
        ulmcp::tool::ToolValue::Boolean(b) => ContextValue::Boolean(b),
        ulmcp::tool::ToolValue::Array(a) => {
            ContextValue::List(a.into_iter().map(tool_value_to_context_value).collect())
        }
        ulmcp::tool::ToolValue::Bytes(b) => ContextValue::Bytes(b),
        ulmcp::tool::ToolValue::Object(m) => ContextValue::Map(
            m.into_iter()
                .map(|(k, v)| (k, tool_value_to_context_value(v)))
                .collect(),
        ),
    }
}

fn apply_transform(
    val: Option<ContextValue>,
    op: &crate::step::TransformOp,
) -> Option<ContextValue> {
    use crate::step::TransformOp;
    let val = val?;
    match op {
        TransformOp::First => match val {
            ContextValue::List(mut l) if !l.is_empty() => Some(l.remove(0)),
            _ => None,
        },
        TransformOp::Last => match val {
            ContextValue::List(mut l) if !l.is_empty() => Some(l.pop()?),
            _ => None,
        },
        TransformOp::Index(i) => match val {
            ContextValue::List(l) => l.into_iter().nth(*i),
            _ => None,
        },
        TransformOp::ToString => Some(ContextValue::String(format!("{:?}", val))),
        TransformOp::Len => match val {
            ContextValue::List(l) => Some(ContextValue::Integer(l.len() as i64)),
            ContextValue::String(s) => Some(ContextValue::Integer(s.len() as i64)),
            _ => None,
        },
        TransformOp::Slice(start, end) => match val {
            ContextValue::List(l) => {
                let end = end.unwrap_or(l.len());
                Some(ContextValue::List(
                    l.into_iter().skip(*start).take(end - start).collect(),
                ))
            }
            ContextValue::String(s) => {
                let end = end.unwrap_or(s.len());
                Some(ContextValue::String(
                    s[*start..end.min(s.len())].to_string(),
                ))
            }
            _ => None,
        },
    }
}

fn chrono_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::{Input, Step};
    use crate::workflow::{Flow, FlowInput};
    use ulmcp::registry::Registry;
    use ulmcp::tool::*;

    fn test_registry() -> Registry {
        let mut reg = Registry::new();

        reg.register_tool(
            ToolDef::new("echo", "Echo input").param(
                "text",
                "Text to echo",
                ParamType::String,
                true,
            ),
            Box::new(|call| ToolResult {
                call_id: call.call_id.clone(),
                status: ToolStatus::Success,
                output: ToolValue::String(
                    call.arguments
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                ),
                error: None,
                tokens_used: Some(10),
                latency_ms: None,
            }),
        );

        reg.register_tool(
            ToolDef::new("add", "Add two numbers")
                .param("a", "First", ParamType::Integer, true)
                .param("b", "Second", ParamType::Integer, true),
            Box::new(|call| {
                let a = call
                    .arguments
                    .get("a")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let b = call
                    .arguments
                    .get("b")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                ToolResult {
                    call_id: call.call_id.clone(),
                    status: ToolStatus::Success,
                    output: ToolValue::Integer(a + b),
                    error: None,
                    tokens_used: Some(5),
                    latency_ms: None,
                }
            }),
        );

        reg.register_tool(
            ToolDef::new("fail", "Always fails"),
            Box::new(|call| ToolResult {
                call_id: call.call_id.clone(),
                status: ToolStatus::Error,
                output: ToolValue::Null,
                error: Some("intentional failure".into()),
                tokens_used: None,
                latency_ms: None,
            }),
        );

        reg
    }

    #[test]
    fn simple_pipeline() {
        let flow = Flow::pipeline("simple")
            .step(
                Step::tool("s1")
                    .tool("echo")
                    .input_literal("text", "hello world")
                    .build(),
            )
            .build()
            .unwrap();

        let mut runner = FlowRunner::new(test_registry());
        let result = runner.run(flow, FlowInput::new()).unwrap();

        assert_eq!(result.status, FlowStatus::Succeeded);
        assert_eq!(result.steps_completed, 1);
        assert_eq!(result.steps_failed, 0);
        assert_eq!(result.get_str("s1.output"), Some("hello world"));
    }

    #[test]
    fn pipeline_with_deps() {
        let flow = Flow::pipeline("chain")
            .step(
                Step::tool("s1")
                    .tool("echo")
                    .input_literal("text", "step1_output")
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
            .unwrap();

        let mut runner = FlowRunner::new(test_registry());
        let result = runner.run(flow, FlowInput::new()).unwrap();

        assert_eq!(result.steps_completed, 2);
        assert_eq!(result.get_str("s2.output"), Some("step1_output"));
    }

    #[test]
    fn flow_with_input_vars() {
        let flow = Flow::pipeline("with_vars")
            .step(
                Step::tool("s1")
                    .tool("echo")
                    .input("text", Input::from_var("task"))
                    .build(),
            )
            .build()
            .unwrap();

        let mut runner = FlowRunner::new(test_registry());
        let result = runner
            .run(flow, FlowInput::new().var("task", "refactor auth"))
            .unwrap();

        assert_eq!(result.get_str("s1.output"), Some("refactor auth"));
    }

    #[test]
    fn flow_step_failure_propagates() {
        let flow = Flow::pipeline("fail_flow")
            .step(Step::tool("bad").tool("fail").build())
            .build()
            .unwrap();

        let mut runner = FlowRunner::new(test_registry());
        let result = runner.run(flow, FlowInput::new());
        assert!(result.is_err());
    }

    #[test]
    fn flow_skip_on_error() {
        let flow = Flow::pipeline("skip_flow")
            .step(Step::tool("bad").tool("fail").skip_on_error().build())
            .step(
                Step::tool("good")
                    .tool("echo")
                    .input_literal("text", "recovered")
                    .depends_on("bad")
                    .build(),
            )
            .build()
            .unwrap();

        let mut runner = FlowRunner::new(test_registry());
        let result = runner.run(flow, FlowInput::new()).unwrap();
        assert_eq!(result.steps_skipped, 1);
        assert_eq!(result.steps_completed, 1);
        assert_eq!(result.get_str("good.output"), Some("recovered"));
    }

    #[test]
    fn numeric_tool() {
        let flow = Flow::pipeline("math")
            .step(
                Step::tool("add")
                    .tool("add")
                    .input_literal("a", 3i64)
                    .input_literal("b", 4i64)
                    .build(),
            )
            .build()
            .unwrap();

        let mut runner = FlowRunner::new(test_registry());
        let result = runner.run(flow, FlowInput::new()).unwrap();
        assert_eq!(result.get("add.output"), Some(&ContextValue::Integer(7)));
    }

    #[test]
    fn events_emitted() {
        use std::sync::{Arc, Mutex};
        let events = Arc::new(Mutex::new(Vec::<String>::new()));
        let events_clone = Arc::clone(&events);

        let flow = Flow::pipeline("events")
            .step(
                Step::tool("s1")
                    .tool("echo")
                    .input_literal("text", "hi")
                    .build(),
            )
            .build()
            .unwrap();

        let mut runner = FlowRunner::new(test_registry());
        runner.on_event(move |e| {
            events_clone.lock().unwrap().push(e.kind.to_string());
        });
        runner.run(flow, FlowInput::new()).unwrap();

        let emitted = events.lock().unwrap();
        assert!(emitted.contains(&"flow.started".to_string()));
        assert!(emitted.contains(&"step.succeeded".to_string()));
        assert!(emitted.contains(&"flow.succeeded".to_string()));
    }

    #[test]
    fn telemetry_recorded() {
        let flow = Flow::pipeline("tel")
            .step(
                Step::tool("s1")
                    .tool("echo")
                    .input_literal("text", "x")
                    .build(),
            )
            .step(
                Step::tool("s2")
                    .tool("echo")
                    .input_literal("text", "y")
                    .depends_on("s1")
                    .build(),
            )
            .build()
            .unwrap();

        let mut runner = FlowRunner::new(test_registry());
        runner.run(flow, FlowInput::new()).unwrap();

        let summary = runner.telemetry_summary();
        assert_eq!(summary.total_spans, 2);
        assert_eq!(summary.error_spans, 0);
    }

    #[test]
    fn agent_step_renders_prompt() {
        let flow = Flow::pipeline("agent_flow")
            .step(Step::agent(
                "analyze",
                "Review {{task}} for security issues",
            ))
            .build()
            .unwrap();

        let mut runner = FlowRunner::new(test_registry());
        let result = runner
            .run(flow, FlowInput::new().var("task", "auth code"))
            .unwrap();

        assert_eq!(result.steps_completed, 1);
        assert_eq!(
            result.get_str("analyze.output"),
            Some("Review auth code for security issues")
        );
    }
}
