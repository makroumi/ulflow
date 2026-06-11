//! ulflow: The agentic AI orchestration engine.
//!
//! Native Rust. Zero compromises. Vertically integrated with the ULMEN ecosystem.
//!
//! Faster than LangChain. More capable than LangGraph. Fully persistent.
//! Built on ulmen-core, uldb, ulmp, and ulmcp.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use ulflow::prelude::*;
//!
//! let flow = Flow::pipeline("my_flow")
//!     .tool("search", "code_search", inputs![("query", "validate token")])
//!     .agent("analyze", "Review: {{search.output}}")
//!     .build();
//!
//! let result = FlowRunner::new(registry).run(flow).await?;
//! ```
//!
//! Copyright (c) 2026 El Mehdi Makroumi. All rights reserved.
//! Licensed under BSL-1.1.

#![forbid(unsafe_code)]

pub mod agent;
pub mod checkpoint;
pub mod context;
pub mod error;
pub mod event;
pub mod json_flow;
pub mod llm;
pub mod memory;
pub mod runner;
pub mod step;
pub mod telemetry;
pub mod workflow;

/// Re-export everything needed to build and run workflows.
pub mod prelude {
    pub use crate::agent::{AgentCall, AgentResult};
    pub use crate::context::{ContextBudget, ExecutionContext};
    pub use crate::error::{FlowError, StepError};
    pub use crate::event::{EventKind, FlowEvent};
    pub use crate::json_flow::flow_from_json;
    pub use crate::llm::{ChatRequest, ChatResponse, Message, Role, LLM};
    pub use crate::memory::{Memory, MemoryScope};
    pub use crate::runner::FlowRunner;
    pub use crate::step::{Input, Step, StepKind, StepResult, StepStatus};
    pub use crate::telemetry::{Span, SpanStatus};
    pub use crate::workflow::{Flow, FlowBuilder, FlowInput, FlowOutput, FlowStatus};
}
