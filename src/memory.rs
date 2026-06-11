//! Agent memory: session-scoped and persistent.

use crate::context::ContextValue;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    Session,
    Persistent,
    Shared,
}

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub key: String,
    pub value: ContextValue,
    pub scope: MemoryScope,
    pub confidence: f64,
    pub created_at_ms: u64,
    pub ttl_ms: Option<u64>,
}

impl MemoryEntry {
    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.ttl_ms
            .map(|ttl| now_ms > self.created_at_ms + ttl)
            .unwrap_or(false)
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub struct Memory {
    entries: HashMap<String, MemoryEntry>,
}

impl Memory {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn store(
        &mut self,
        key: impl Into<String>,
        value: impl Into<ContextValue>,
        scope: MemoryScope,
        confidence: f64,
    ) {
        let key = key.into();
        self.entries.insert(
            key.clone(),
            MemoryEntry {
                key,
                value: value.into(),
                scope,
                confidence,
                created_at_ms: now_ms(),
                ttl_ms: None,
            },
        );
    }

    pub fn store_with_ttl(
        &mut self,
        key: impl Into<String>,
        value: impl Into<ContextValue>,
        scope: MemoryScope,
        confidence: f64,
        ttl_ms: u64,
    ) {
        let key = key.into();
        self.entries.insert(
            key.clone(),
            MemoryEntry {
                key,
                value: value.into(),
                scope,
                confidence,
                created_at_ms: now_ms(),
                ttl_ms: Some(ttl_ms),
            },
        );
    }

    pub fn get(&self, key: &str) -> Option<&MemoryEntry> {
        self.entries.get(key).filter(|e| !e.is_expired(now_ms()))
    }

    pub fn get_value(&self, key: &str) -> Option<&ContextValue> {
        self.get(key).map(|e| &e.value)
    }
    pub fn remove(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }
    pub fn by_scope(&self, scope: MemoryScope) -> Vec<&MemoryEntry> {
        self.entries.values().filter(|e| e.scope == scope).collect()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn evict_expired(&mut self) {
        let t = now_ms();
        self.entries.retain(|_, e| !e.is_expired(t));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn store_and_get() {
        let mut mem = Memory::new();
        mem.store("k1", "v1", MemoryScope::Session, 0.9);
        assert_eq!(mem.get_value("k1").and_then(|v| v.as_str()), Some("v1"));
    }

    #[test]
    fn ttl_expiry() {
        let mut mem = Memory::new();
        mem.store_with_ttl("exp", "v", MemoryScope::Session, 1.0, 1);
        thread::sleep(Duration::from_millis(5));
        assert!(mem.get("exp").is_none());
    }

    #[test]
    fn by_scope() {
        let mut mem = Memory::new();
        mem.store("s1", "v", MemoryScope::Session, 1.0);
        mem.store("p1", "v", MemoryScope::Persistent, 1.0);
        assert_eq!(mem.by_scope(MemoryScope::Session).len(), 1);
        assert_eq!(mem.by_scope(MemoryScope::Persistent).len(), 1);
    }

    #[test]
    fn evict_expired() {
        let mut mem = Memory::new();
        mem.store_with_ttl("old", "v", MemoryScope::Session, 1.0, 1);
        mem.store("fresh", "v", MemoryScope::Session, 1.0);
        thread::sleep(Duration::from_millis(5));
        mem.evict_expired();
        assert_eq!(mem.len(), 1);
    }
}
