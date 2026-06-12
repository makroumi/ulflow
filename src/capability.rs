//! Capability-based security kernel for AI agents.
//!
//! Every agent gets a capability set at creation time.
//! Every operation checks capabilities before execution.
//! No capability = denied. No exceptions. No bypass.
//!
//! This is the security boundary of the AI runtime.
//!
//! ```rust,ignore
//! let caps = Capabilities::new("agent_x")
//!     .allow_tool("code_search")
//!     .allow_tool("file_read")
//!     .deny_tool("file_write")
//!     .allow_db_read()
//!     .deny_db_write()
//!     .allow_llm("groq")
//!     .token_budget(5000)
//!     .namespace("repo-a");
//!
//! let agent = SecureAgent::new("agent_x", caps, llm, registry);
//! agent.call_tool("code_search", args)?;   // OK
//! agent.call_tool("file_write", args)?;     // DENIED
//! ```

use std::collections::HashSet;
use std::fmt;

/// A single capability.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Cap {
    /// Can call a specific tool.
    Tool(String),
    /// Can call any tool.
    ToolAll,
    /// Can read from the database.
    DbRead,
    /// Can write to the database.
    DbWrite,
    /// Can search the database.
    DbSearch,
    /// Can access a specific namespace.
    Namespace(String),
    /// Can use a specific LLM provider.
    Llm(String),
    /// Can use any LLM.
    LlmAll,
    /// Can create sessions.
    SessionCreate,
    /// Can read session history.
    SessionRead,
    /// Can execute workflows.
    WorkflowExecute,
    /// Can register new workflows.
    WorkflowRegister,
    /// Can read metrics.
    AdminMetrics,
    /// Can read logs.
    AdminLogs,
}

impl fmt::Display for Cap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cap::Tool(name) => write!(f, "tool:{}", name),
            Cap::ToolAll => write!(f, "tool:*"),
            Cap::DbRead => write!(f, "db:read"),
            Cap::DbWrite => write!(f, "db:write"),
            Cap::DbSearch => write!(f, "db:search"),
            Cap::Namespace(ns) => write!(f, "namespace:{}", ns),
            Cap::Llm(provider) => write!(f, "llm:{}", provider),
            Cap::LlmAll => write!(f, "llm:*"),
            Cap::SessionCreate => write!(f, "session:create"),
            Cap::SessionRead => write!(f, "session:read"),
            Cap::WorkflowExecute => write!(f, "workflow:execute"),
            Cap::WorkflowRegister => write!(f, "workflow:register"),
            Cap::AdminMetrics => write!(f, "admin:metrics"),
            Cap::AdminLogs => write!(f, "admin:logs"),
        }
    }
}

impl Cap {
    /// Parse a capability from a string: "tool:code_search", "db:read", "llm:groq", etc.
    pub fn parse(s: &str) -> Option<Self> {
        let (prefix, value) = s.split_once(':')?;
        match prefix {
            "tool" => {
                if value == "*" {
                    Some(Cap::ToolAll)
                } else {
                    Some(Cap::Tool(value.into()))
                }
            }
            "db" => match value {
                "read" => Some(Cap::DbRead),
                "write" => Some(Cap::DbWrite),
                "search" => Some(Cap::DbSearch),
                _ => None,
            },
            "namespace" => Some(Cap::Namespace(value.into())),
            "llm" => {
                if value == "*" {
                    Some(Cap::LlmAll)
                } else {
                    Some(Cap::Llm(value.into()))
                }
            }
            "session" => match value {
                "create" => Some(Cap::SessionCreate),
                "read" => Some(Cap::SessionRead),
                _ => None,
            },
            "workflow" => match value {
                "execute" => Some(Cap::WorkflowExecute),
                "register" => Some(Cap::WorkflowRegister),
                _ => None,
            },
            "admin" => match value {
                "metrics" => Some(Cap::AdminMetrics),
                "logs" => Some(Cap::AdminLogs),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Access denied error.
#[derive(Debug, Clone)]
pub struct AccessDenied {
    pub agent: String,
    pub capability: String,
    pub message: String,
}

impl fmt::Display for AccessDenied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "access denied: agent {:?} lacks capability {} ({})",
            self.agent, self.capability, self.message
        )
    }
}

impl std::error::Error for AccessDenied {}

/// Capability set for an agent. Immutable after creation.
#[derive(Debug, Clone)]
pub struct Capabilities {
    pub agent_name: String,
    caps: HashSet<Cap>,
    pub token_budget: Option<usize>,
    pub tokens_used: usize,
}

impl Capabilities {
    /// Create an empty capability set (no permissions at all).
    pub fn new(agent_name: impl Into<String>) -> Self {
        Self {
            agent_name: agent_name.into(),
            caps: HashSet::new(),
            token_budget: None,
            tokens_used: 0,
        }
    }

    /// Create a superuser capability set (all permissions).
    pub fn superuser(agent_name: impl Into<String>) -> Self {
        let mut caps = HashSet::new();
        caps.insert(Cap::ToolAll);
        caps.insert(Cap::DbRead);
        caps.insert(Cap::DbWrite);
        caps.insert(Cap::DbSearch);
        caps.insert(Cap::LlmAll);
        caps.insert(Cap::SessionCreate);
        caps.insert(Cap::SessionRead);
        caps.insert(Cap::WorkflowExecute);
        caps.insert(Cap::WorkflowRegister);
        caps.insert(Cap::AdminMetrics);
        caps.insert(Cap::AdminLogs);
        Self {
            agent_name: agent_name.into(),
            caps,
            token_budget: None,
            tokens_used: 0,
        }
    }

    /// Create from a list of capability strings.
    pub fn from_strings(agent_name: impl Into<String>, caps: &[&str]) -> Self {
        let mut set = HashSet::new();
        for s in caps {
            if let Some(cap) = Cap::parse(s) {
                set.insert(cap);
            }
        }
        Self {
            agent_name: agent_name.into(),
            caps: set,
            token_budget: None,
            tokens_used: 0,
        }
    }

    // Builder methods

    pub fn allow_tool(mut self, name: impl Into<String>) -> Self {
        self.caps.insert(Cap::Tool(name.into()));
        self
    }

    pub fn allow_all_tools(mut self) -> Self {
        self.caps.insert(Cap::ToolAll);
        self
    }

    pub fn allow_db_read(mut self) -> Self {
        self.caps.insert(Cap::DbRead);
        self
    }

    pub fn allow_db_write(mut self) -> Self {
        self.caps.insert(Cap::DbWrite);
        self
    }

    pub fn allow_db_search(mut self) -> Self {
        self.caps.insert(Cap::DbSearch);
        self
    }

    pub fn allow_llm(mut self, provider: impl Into<String>) -> Self {
        self.caps.insert(Cap::Llm(provider.into()));
        self
    }

    pub fn allow_all_llm(mut self) -> Self {
        self.caps.insert(Cap::LlmAll);
        self
    }

    pub fn allow_sessions(mut self) -> Self {
        self.caps.insert(Cap::SessionCreate);
        self.caps.insert(Cap::SessionRead);
        self
    }

    pub fn allow_workflows(mut self) -> Self {
        self.caps.insert(Cap::WorkflowExecute);
        self
    }

    pub fn allow_admin(mut self) -> Self {
        self.caps.insert(Cap::AdminMetrics);
        self.caps.insert(Cap::AdminLogs);
        self
    }

    pub fn namespace(mut self, ns: impl Into<String>) -> Self {
        self.caps.insert(Cap::Namespace(ns.into()));
        self
    }

    pub fn token_budget(mut self, budget: usize) -> Self {
        self.token_budget = Some(budget);
        self
    }

    pub fn deny_tool(mut self, name: &str) -> Self {
        self.caps.remove(&Cap::Tool(name.into()));
        self
    }

    // Check methods

    /// Check if a specific capability is granted.
    pub fn has(&self, cap: &Cap) -> bool {
        self.caps.contains(cap)
    }

    /// Check if the agent can call a specific tool.
    pub fn can_call_tool(&self, tool_name: &str) -> bool {
        self.caps.contains(&Cap::ToolAll) || self.caps.contains(&Cap::Tool(tool_name.into()))
    }

    /// Check if the agent can use a specific LLM provider.
    pub fn can_use_llm(&self, provider: &str) -> bool {
        self.caps.contains(&Cap::LlmAll) || self.caps.contains(&Cap::Llm(provider.into()))
    }

    /// Check if the agent can access a specific namespace.
    pub fn can_access_namespace(&self, ns: &str) -> bool {
        self.caps.contains(&Cap::Namespace(ns.into()))
    }

    /// Check and enforce a capability. Returns Err(AccessDenied) if denied.
    pub fn require(&self, cap: &Cap) -> Result<(), AccessDenied> {
        // Tool check: specific or wildcard
        let granted = match cap {
            Cap::Tool(name) => self.can_call_tool(name),
            Cap::Llm(provider) => self.can_use_llm(provider),
            Cap::Namespace(ns) => self.can_access_namespace(ns),
            other => self.caps.contains(other),
        };

        if granted {
            Ok(())
        } else {
            Err(AccessDenied {
                agent: self.agent_name.clone(),
                capability: cap.to_string(),
                message: format!("agent {:?} does not have {}", self.agent_name, cap),
            })
        }
    }

    /// Check and enforce db:read capability.
    pub fn require_db_read(&self) -> Result<(), AccessDenied> {
        self.require(&Cap::DbRead)
    }

    /// Check and enforce db:write capability.
    pub fn require_db_write(&self) -> Result<(), AccessDenied> {
        self.require(&Cap::DbWrite)
    }

    /// Check and enforce db:search capability.
    pub fn require_db_search(&self) -> Result<(), AccessDenied> {
        self.require(&Cap::DbSearch)
    }

    /// Check and enforce session:create capability.
    pub fn require_session_create(&self) -> Result<(), AccessDenied> {
        self.require(&Cap::SessionCreate)
    }

    /// Check and enforce session:read capability.
    pub fn require_session_read(&self) -> Result<(), AccessDenied> {
        self.require(&Cap::SessionRead)
    }

    /// Check and enforce workflow:execute capability.
    pub fn require_workflow_execute(&self) -> Result<(), AccessDenied> {
        self.require(&Cap::WorkflowExecute)
    }

    /// Check and enforce workflow:register capability.
    pub fn require_workflow_register(&self) -> Result<(), AccessDenied> {
        self.require(&Cap::WorkflowRegister)
    }

    /// Check and enforce namespace access.
    pub fn require_namespace(&self, ns: &str) -> Result<(), AccessDenied> {
        self.require(&Cap::Namespace(ns.into()))
    }

    /// Check and enforce tool call capability.
    pub fn require_tool(&self, tool_name: &str) -> Result<(), AccessDenied> {
        self.require(&Cap::Tool(tool_name.into()))
    }

    /// Check and enforce LLM capability.
    pub fn require_llm(&self, provider: &str) -> Result<(), AccessDenied> {
        self.require(&Cap::Llm(provider.into()))
    }

    /// Check and consume tokens from budget. Returns Err if over budget.
    pub fn use_tokens(&mut self, n: usize) -> Result<(), AccessDenied> {
        if let Some(budget) = self.token_budget {
            if self.tokens_used + n > budget {
                return Err(AccessDenied {
                    agent: self.agent_name.clone(),
                    capability: format!("llm:budget:{}", budget),
                    message: format!(
                        "token budget exceeded: used {} + {} > budget {}",
                        self.tokens_used, n, budget
                    ),
                });
            }
        }
        self.tokens_used += n;
        Ok(())
    }

    /// Remaining token budget.
    pub fn tokens_remaining(&self) -> Option<usize> {
        self.token_budget
            .map(|b| b.saturating_sub(self.tokens_used))
    }

    /// List all granted capabilities.
    pub fn list(&self) -> Vec<String> {
        let mut caps: Vec<String> = self.caps.iter().map(|c| c.to_string()).collect();
        caps.sort();
        if let Some(budget) = self.token_budget {
            caps.push(format!("llm:budget:{}", budget));
        }
        caps
    }

    /// Total number of capabilities.
    pub fn count(&self) -> usize {
        self.caps.len()
    }
}

// -----------------------------------------------------------------------
// Predefined capability profiles
// -----------------------------------------------------------------------

impl Capabilities {
    /// Read-only agent: can search and read, nothing else.
    pub fn read_only(agent_name: impl Into<String>) -> Self {
        Self::new(agent_name)
            .allow_tool("code_search")
            .allow_tool("file_read")
            .allow_db_read()
            .allow_db_search()
            .allow_all_llm()
    }

    /// Writer agent: can read, search, and write.
    pub fn writer(agent_name: impl Into<String>) -> Self {
        Self::read_only(agent_name)
            .allow_tool("file_write")
            .allow_db_write()
    }

    /// Analyst agent: can read and use LLM, but not write.
    pub fn analyst(agent_name: impl Into<String>) -> Self {
        Self::read_only(agent_name)
            .allow_sessions()
            .allow_workflows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_caps_deny_everything() {
        let caps = Capabilities::new("agent_empty");
        assert!(!caps.can_call_tool("code_search"));
        assert!(!caps.can_use_llm("groq"));
        assert!(caps.require_tool("code_search").is_err());
    }

    #[test]
    fn specific_tool_allowed() {
        let caps = Capabilities::new("agent_x")
            .allow_tool("code_search")
            .allow_tool("file_read");
        assert!(caps.can_call_tool("code_search"));
        assert!(caps.can_call_tool("file_read"));
        assert!(!caps.can_call_tool("file_write"));
        assert!(caps.require_tool("file_write").is_err());
    }

    #[test]
    fn wildcard_tool() {
        let caps = Capabilities::new("admin").allow_all_tools();
        assert!(caps.can_call_tool("anything"));
        assert!(caps.can_call_tool("even_this"));
    }

    #[test]
    fn deny_after_allow() {
        let caps = Capabilities::new("agent")
            .allow_all_tools()
            .deny_tool("file_write");
        // deny_tool removes the specific tool, but ToolAll still matches
        assert!(caps.can_call_tool("file_write")); // ToolAll overrides
    }

    #[test]
    fn specific_llm() {
        let caps = Capabilities::new("agent")
            .allow_llm("groq")
            .allow_llm("ollama");
        assert!(caps.can_use_llm("groq"));
        assert!(caps.can_use_llm("ollama"));
        assert!(!caps.can_use_llm("openai"));
    }

    #[test]
    fn wildcard_llm() {
        let caps = Capabilities::new("agent").allow_all_llm();
        assert!(caps.can_use_llm("anything"));
    }

    #[test]
    fn token_budget() {
        let mut caps = Capabilities::new("agent").token_budget(1000);
        assert_eq!(caps.tokens_remaining(), Some(1000));
        assert!(caps.use_tokens(500).is_ok());
        assert_eq!(caps.tokens_remaining(), Some(500));
        assert!(caps.use_tokens(600).is_err()); // over budget
        assert_eq!(caps.tokens_used, 500); // unchanged
    }

    #[test]
    fn no_budget_unlimited() {
        let mut caps = Capabilities::new("agent");
        assert_eq!(caps.tokens_remaining(), None);
        assert!(caps.use_tokens(1_000_000).is_ok()); // no limit
    }

    #[test]
    fn namespace_isolation() {
        let caps = Capabilities::new("agent")
            .namespace("repo-a")
            .namespace("repo-b");
        assert!(caps.can_access_namespace("repo-a"));
        assert!(caps.can_access_namespace("repo-b"));
        assert!(!caps.can_access_namespace("repo-c"));
    }

    #[test]
    fn superuser_has_everything() {
        let caps = Capabilities::superuser("admin");
        assert!(caps.can_call_tool("anything"));
        assert!(caps.can_use_llm("anything"));
        assert!(caps.has(&Cap::DbRead));
        assert!(caps.has(&Cap::DbWrite));
        assert!(caps.has(&Cap::AdminLogs));
    }

    #[test]
    fn from_strings() {
        let caps = Capabilities::from_strings(
            "agent",
            &[
                "tool:code_search",
                "tool:file_read",
                "db:read",
                "db:search",
                "llm:groq",
                "namespace:repo-a",
            ],
        );
        assert!(caps.can_call_tool("code_search"));
        assert!(caps.can_call_tool("file_read"));
        assert!(!caps.can_call_tool("file_write"));
        assert!(caps.has(&Cap::DbRead));
        assert!(!caps.has(&Cap::DbWrite));
        assert!(caps.can_use_llm("groq"));
        assert!(!caps.can_use_llm("openai"));
    }

    #[test]
    fn list_caps() {
        let caps = Capabilities::new("agent")
            .allow_tool("search")
            .allow_db_read()
            .token_budget(5000);
        let list = caps.list();
        assert!(list.contains(&"tool:search".to_string()));
        assert!(list.contains(&"db:read".to_string()));
        assert!(list.contains(&"llm:budget:5000".to_string()));
    }

    #[test]
    fn predefined_read_only() {
        let caps = Capabilities::read_only("reader");
        assert!(caps.can_call_tool("code_search"));
        assert!(caps.can_call_tool("file_read"));
        assert!(!caps.can_call_tool("file_write"));
        assert!(caps.has(&Cap::DbRead));
        assert!(!caps.has(&Cap::DbWrite));
    }

    #[test]
    fn predefined_writer() {
        let caps = Capabilities::writer("writer");
        assert!(caps.can_call_tool("file_write"));
        assert!(caps.has(&Cap::DbWrite));
    }

    #[test]
    fn predefined_analyst() {
        let caps = Capabilities::analyst("analyst");
        assert!(caps.can_call_tool("code_search"));
        assert!(!caps.can_call_tool("file_write"));
        assert!(caps.has(&Cap::SessionCreate));
        assert!(caps.has(&Cap::WorkflowExecute));
    }

    #[test]
    fn access_denied_display() {
        let caps = Capabilities::new("agent_x");
        let err = caps.require_tool("file_write").unwrap_err();
        assert!(err.to_string().contains("agent_x"));
        assert!(err.to_string().contains("file_write"));
    }

    #[test]
    fn cap_parse_roundtrip() {
        let cases = &[
            "tool:code_search",
            "tool:*",
            "db:read",
            "db:write",
            "db:search",
            "namespace:repo-a",
            "llm:groq",
            "llm:*",
            "session:create",
            "session:read",
            "workflow:execute",
            "workflow:register",
            "admin:metrics",
            "admin:logs",
        ];
        for s in cases {
            let cap = Cap::parse(s).unwrap_or_else(|| panic!("failed to parse: {}", s));
            assert_eq!(cap.to_string(), *s, "roundtrip failed for {}", s);
        }
    }

    #[test]
    fn cap_parse_invalid() {
        assert!(Cap::parse("invalid").is_none());
        assert!(Cap::parse("unknown:value").is_none());
        assert!(Cap::parse("db:invalid").is_none());
    }

    #[test]
    fn require_db_read_denied() {
        let caps = Capabilities::new("agent");
        assert!(caps.require_db_read().is_err());
    }

    #[test]
    fn require_db_read_allowed() {
        let caps = Capabilities::new("agent").allow_db_read();
        assert!(caps.require_db_read().is_ok());
    }

    #[test]
    fn require_db_write_denied() {
        let caps = Capabilities::new("agent").allow_db_read();
        assert!(caps.require_db_write().is_err());
    }

    #[test]
    fn require_session_create() {
        let caps = Capabilities::new("agent").allow_sessions();
        assert!(caps.require_session_create().is_ok());
        assert!(caps.require_session_read().is_ok());
    }

    #[test]
    fn require_workflow_execute() {
        let caps = Capabilities::new("agent").allow_workflows();
        assert!(caps.require_workflow_execute().is_ok());
        assert!(caps.require_workflow_register().is_err()); // allow_workflows only adds execute
    }

    #[test]
    fn require_namespace_isolation() {
        let caps = Capabilities::new("agent")
            .namespace("repo-a")
            .namespace("repo-b");
        assert!(caps.require_namespace("repo-a").is_ok());
        assert!(caps.require_namespace("repo-b").is_ok());
        assert!(caps.require_namespace("repo-c").is_err());
    }
}
