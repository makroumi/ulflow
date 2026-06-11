//! LLM provider implementations.
//!
//! Each provider translates the universal ChatRequest into
//! the provider-specific HTTP request and parses the response.

use crate::llm::types::*;

/// Send a chat request to OpenAI-compatible API.
/// Works with: OpenAI, Azure OpenAI, Groq, Together, Fireworks, Perplexity, vLLM, LiteLLM.
pub fn call_openai(
    config: &ProviderConfig,
    request: &ChatRequest,
) -> Result<ChatResponse, LLMError> {
    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));

    let mut messages_json: Vec<serde_json::Value> = Vec::new();

    // OpenAI uses system message in the messages array
    if let Some(ref sys) = request.system {
        messages_json.push(serde_json::json!({"role": "system", "content": sys}));
    }

    for msg in &request.messages {
        messages_json.push(serde_json::json!({
            "role": msg.role.to_string(),
            "content": msg.content,
        }));
    }

    let mut body = serde_json::json!({
        "model": request.model,
        "messages": messages_json,
    });

    if let Some(t) = request.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(mt) = request.max_tokens {
        body["max_tokens"] = serde_json::json!(mt);
    }
    if !request.stop.is_empty() {
        body["stop"] = serde_json::json!(request.stop);
    }

    let mut req = ureq::post(&url).set("Content-Type", "application/json");

    if let Some(ref key) = config.api_key {
        req = req.set("Authorization", &format!("Bearer {}", key));
    }

    for (k, v) in &config.headers {
        req = req.set(k, v);
    }

    let resp = req.send_json(&body).map_err(|e| LLMError {
        message: e.to_string(),
        status_code: None,
        provider: config.name.clone(),
    })?;

    let status = resp.status();
    let body_str = resp.into_string().map_err(|e| LLMError {
        message: format!("read body: {}", e),
        status_code: Some(status),
        provider: config.name.clone(),
    })?;

    if status != 200 {
        return Err(LLMError {
            message: body_str,
            status_code: Some(status),
            provider: config.name.clone(),
        });
    }

    let json: serde_json::Value = serde_json::from_str(&body_str).map_err(|e| LLMError {
        message: format!("parse json: {}", e),
        status_code: Some(status),
        provider: config.name.clone(),
    })?;

    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let finish = match json["choices"][0]["finish_reason"].as_str() {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::MaxTokens,
        _ => FinishReason::Stop,
    };
    let input_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize;
    let output_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize;
    let model = json["model"].as_str().unwrap_or(&request.model).to_string();

    Ok(ChatResponse {
        content,
        model,
        input_tokens,
        output_tokens,
        finish_reason: finish,
    })
}

/// Send a chat request to Anthropic API.
pub fn call_anthropic(
    config: &ProviderConfig,
    request: &ChatRequest,
) -> Result<ChatResponse, LLMError> {
    let url = format!("{}/messages", config.base_url.trim_end_matches('/'));

    let messages_json: Vec<serde_json::Value> = request
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(|m| serde_json::json!({"role": m.role.to_string(), "content": m.content}))
        .collect();

    let mut body = serde_json::json!({
        "model": request.model,
        "messages": messages_json,
        "max_tokens": request.max_tokens.unwrap_or(4096),
    });

    // Anthropic uses top-level system field
    let system_text = request.system.clone().or_else(|| {
        request
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.clone())
    });
    if let Some(sys) = system_text {
        body["system"] = serde_json::json!(sys);
    }

    if let Some(t) = request.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if !request.stop.is_empty() {
        body["stop_sequences"] = serde_json::json!(request.stop);
    }

    let mut req = ureq::post(&url)
        .set("Content-Type", "application/json")
        .set("anthropic-version", "2023-06-01");

    if let Some(ref key) = config.api_key {
        req = req.set("x-api-key", key);
    }

    for (k, v) in &config.headers {
        req = req.set(k, v);
    }

    let resp = req.send_json(&body).map_err(|e| LLMError {
        message: e.to_string(),
        status_code: None,
        provider: config.name.clone(),
    })?;

    let status = resp.status();
    let body_str = resp.into_string().map_err(|e| LLMError {
        message: format!("read body: {}", e),
        status_code: Some(status),
        provider: config.name.clone(),
    })?;

    if status != 200 {
        return Err(LLMError {
            message: body_str,
            status_code: Some(status),
            provider: config.name.clone(),
        });
    }

    let json: serde_json::Value = serde_json::from_str(&body_str).map_err(|e| LLMError {
        message: format!("parse json: {}", e),
        status_code: Some(status),
        provider: config.name.clone(),
    })?;

    let content = json["content"][0]["text"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let finish = match json["stop_reason"].as_str() {
        Some("end_turn") => FinishReason::Stop,
        Some("max_tokens") => FinishReason::MaxTokens,
        _ => FinishReason::Stop,
    };
    let input_tokens = json["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize;
    let output_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0) as usize;
    let model = json["model"].as_str().unwrap_or(&request.model).to_string();

    Ok(ChatResponse {
        content,
        model,
        input_tokens,
        output_tokens,
        finish_reason: finish,
    })
}

/// Send a chat request to Ollama (local).
pub fn call_ollama(
    config: &ProviderConfig,
    request: &ChatRequest,
) -> Result<ChatResponse, LLMError> {
    let url = format!("{}/chat", config.base_url.trim_end_matches('/'));

    let mut messages_json: Vec<serde_json::Value> = Vec::new();
    if let Some(ref sys) = request.system {
        messages_json.push(serde_json::json!({"role": "system", "content": sys}));
    }
    for msg in &request.messages {
        messages_json
            .push(serde_json::json!({"role": msg.role.to_string(), "content": msg.content}));
    }

    let body = serde_json::json!({
        "model": request.model,
        "messages": messages_json,
        "stream": false,
    });

    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_json(&body)
        .map_err(|e| LLMError {
            message: e.to_string(),
            status_code: None,
            provider: config.name.clone(),
        })?;

    let status = resp.status();
    let body_str = resp.into_string().map_err(|e| LLMError {
        message: format!("read body: {}", e),
        status_code: Some(status),
        provider: config.name.clone(),
    })?;

    if status != 200 {
        return Err(LLMError {
            message: body_str,
            status_code: Some(status),
            provider: config.name.clone(),
        });
    }

    let json: serde_json::Value = serde_json::from_str(&body_str).map_err(|e| LLMError {
        message: format!("parse json: {}", e),
        status_code: Some(status),
        provider: config.name.clone(),
    })?;

    let content = json["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let input_tokens = json["prompt_eval_count"].as_u64().unwrap_or(0) as usize;
    let output_tokens = json["eval_count"].as_u64().unwrap_or(0) as usize;

    Ok(ChatResponse {
        content,
        model: request.model.clone(),
        input_tokens,
        output_tokens,
        finish_reason: FinishReason::Stop,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_config() {
        let config = ProviderConfig {
            name: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            api_key: Some("sk-test".into()),
            default_model: "gpt-4o".into(),
            headers: Vec::new(),
        };
        assert_eq!(config.name, "openai");
    }

    #[test]
    fn anthropic_config() {
        let config = ProviderConfig {
            name: "anthropic".into(),
            base_url: "https://api.anthropic.com/v1".into(),
            api_key: Some("sk-ant-test".into()),
            default_model: "claude-3-5-sonnet-20241022".into(),
            headers: Vec::new(),
        };
        assert_eq!(config.default_model, "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn ollama_config() {
        let config = ProviderConfig {
            name: "ollama".into(),
            base_url: "http://localhost:11434/api".into(),
            api_key: None,
            default_model: "llama3".into(),
            headers: Vec::new(),
        };
        assert!(config.api_key.is_none());
    }
}
