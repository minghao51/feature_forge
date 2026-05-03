# Observability: structlog + Langfuse + OpenTelemetry

## Philosophy

Every operation in `feature_forge` must be **observable**: logged, traced, and correlated. This enables debugging, cost tracking, and performance optimization.

## Layer 1: Structured Logging with structlog

### Why structlog?
- **2x faster** than standard library logging (benchmarked)
- **JSON in production**, pretty/colorful in development (auto-detected TTY)
- **Context binding** with `contextvars` for async-safe propagation
- **OpenTelemetry integration** for trace correlation

### Configuration

```python
# src/feature_forge/observability/structlog_config.py
import sys
import structlog
from opentelemetry import trace

def add_open_telemetry_spans(logger, method_name, event_dict):
    """Inject current OTel span context into log events."""
    span = trace.get_current_span()
    if not span.is_recording():
        event_dict["span"] = None
        return event_dict
    
    ctx = span.get_span_context()
    event_dict["span"] = {
        "span_id": format(ctx.span_id, "016x"),
        "trace_id": format(ctx.trace_id, "032x"),
    }
    return event_dict

def configure_logging():
    """Configure structlog for JSON (prod) or pretty (dev) output."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_open_telemetry_spans,
    ]
    
    if sys.stderr.isatty():
        # Development: colorful, human-readable
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Production: JSON for log aggregators
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

### Usage Patterns

```python
from structlog.contextvars import bind_contextvars, unbind_contextvars
import structlog

logger = structlog.get_logger()

# Bind context at experiment start
bind_contextvars(
    experiment_id="exp-001",
    dataset="titanic",
    seed=42,
    task="classification",
)

# All subsequent logs include these fields
logger.info("pipeline_started", n_rounds=4)
# → {"event": "pipeline_started", "n_rounds": 4, "experiment_id": "exp-001", ...}

# Bind per-round context
bind_contextvars(round=2)
logger.info("agent_selected", agents=["unary", "cross"])
# → {"event": "agent_selected", "agents": ["unary", "cross"], "round": 2, ...}

# Unbind when done
unbind_contextvars("round")
```

### Canonical Log Lines

Instead of many small logs, bind data incrementally and emit a summary:

```python
logger.info(
    "round_completed",
    round=2,
    n_features_generated=12,
    n_effective_features=5,
    avg_feature_gain=0.032,
    llm_cost_usd=0.15,
    latency_seconds=45.2,
)
```

### Exception Logging

```python
try:
    result = sandbox.execute(code, globals_dict)
except CodeExecutionError:
    logger.exception("sandbox_execution_failed", code_length=len(code))
    # → Includes structured traceback with dict_tracebacks processor
```

---

## Layer 2: LLM Tracing with Langfuse

### Why Langfuse?
- **Hierarchical tracing**: Router → Agent → LLM Call → Tool Execution
- **Cost tracking**: Token usage → USD per call
- **Prompt management**: Versioned prompts, A/B testing
- **Cloud offering**: Zero infrastructure overhead
- **OpenTelemetry**: Native OTel export for correlation with structlog

### Integration Pattern

```python
# src/feature_forge/observability/langfuse_tracer.py
from langfuse import observe
from langfuse import Langfuse

langfuse = Langfuse()

# Wrap agent execution
@observe(name="unary-agent", as_type="agent", capture_input=True)
async def generate_features(self, X, y, context):
    """Generate features with automatic tracing."""
    # Inputs/outputs captured automatically
    # Latency measured automatically
    return features

# Wrap LLM calls
@observe(name="feature-plan-generation", as_type="generation")
async def _generate_plan(self, prompt, model, temperature):
    """LLM call with token/cost tracking."""
    response = await self.llm_client.complete(prompt, model, temperature)
    
    # Update with usage metrics
    # (Langfuse auto-captures if using wrapped OpenAI client)
    return response

# Wrap tool execution
@observe(name="sandbox-execution", as_type="tool")
def execute_code(self, code, globals_dict):
    """Sandboxed execution traced as tool call."""
    return sandbox.execute(code, globals_dict)
```

### Hierarchical Trace Structure

```
Trace: titanic_seed42_malmas_full
├── Span: round_1
│   ├── Span: router_select_agents
│   │   └── Output: ["unary", "cross_compositional"]
│   ├── Span: unary_agent
│   │   ├── Generation: feature_plan
│   │   │   ├── Input: prompt (truncated)
│   │   │   ├── Output: JSON feature plan
│   │   │   ├── Usage: {prompt_tokens: 1200, completion_tokens: 400}
│   │   │   └── Cost: $0.003
│   │   ├── Generation: code_generation
│   │   │   ├── Usage: {prompt_tokens: 800, completion_tokens: 200}
│   │   │   └── Cost: $0.002
│   │   ├── Tool: sandbox_execution
│   │   │   ├── Input: generated Python code
│   │   │   └── Output: {success: true, n_features: 3}
│   │   └── Span: feature_evaluation
│   │       ├── Metric: base_auc = 0.82
│   │       ├── Metric: new_auc = 0.845
│   │       └── Metric: gain = 0.025
│   └── Span: cross_compositional_agent
│       └── ... (same structure)
├── Span: persist_top_features
│   └── Metric: n_persisted = 4
└── Span: round_2
    └── ...
```

### Prompt Management

```python
# Fetch prompt from Langfuse (versioned)
prompt = langfuse.get_prompt("unary-feature-agent", version=3)
messages = prompt.compile(dataset_description=description)

# Link prompt to generation for tracking
with langfuse.start_as_current_observation(
    name="prompted-generation",
    as_type="generation",
    prompt=prompt
) as gen:
    response = await llm_client.complete(messages)
    gen.update(output=response)
```

### Cost Tracking

Langfuse automatically tracks costs when using the wrapped OpenAI client:

```python
from langfuse.openai import openai

# All calls traced with tokens, latency, cost
client = openai.AsyncOpenAI()
response = await client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    name="feature-generation",  # Custom span name
    metadata={"agent": "unary", "round": 2},
)
```

### Multi-Agent Session Tracking

```python
# All agents in a pipeline share a trace_id
@observe()
async def run_pipeline(dataset, config):
    # This creates the root trace
    for round_num in range(config.n_rounds):
        agents = router.select_agents(...)
        for agent in agents:
            await agent.generate(...)  # Child spans auto-linked
```

---

## Layer 3: OpenTelemetry Integration

### Why OpenTelemetry?
- **Vendor-neutral**: Not locked into Langfuse or WandB
- **Standard**: Industry standard for distributed tracing
- **Correlation**: Links logs (structlog) + traces (Langfuse) + metrics (WandB)

### Integration

Langfuse exports OTel traces natively. structlog includes `trace_id` and `span_id` in every log event via the `add_open_telemetry_spans` processor.

**Result:** You can search logs by trace ID, or click from a Langfuse trace to see all associated logs.

---

## Observability Checklist

| Component | structlog | Langfuse | OTel |
|-----------|-----------|----------|------|
| Pipeline start/end | ✅ | ✅ (root trace) | ✅ |
| Router selection | ✅ | ✅ (span) | ✅ |
| Agent execution | ✅ | ✅ (span) | ✅ |
| LLM calls | ✅ | ✅ (generation) | ✅ |
| Code execution | ✅ | ✅ (tool) | ✅ |
| Feature evaluation | ✅ | ✅ (span) | ✅ |
| Memory updates | ✅ | ❌ | ❌ |
| Experiment metrics | ✅ | ❌ | ❌ |
| Errors/exceptions | ✅ | ✅ (error span) | ✅ |

---

## Configuration

```yaml
# config/logging.yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "json"  # json or pretty (auto-detected if not set)
  
langfuse:
  enabled: true
  public_key: "${LANGFUSE_PUBLIC_KEY}"  # From .env
  secret_key: "${LANGFUSE_SECRET_KEY}"  # From .env
  host: "https://cloud.langfuse.com"
  
opentelemetry:
  enabled: true
  exporter: "langfuse"  # Langfuse acts as OTel backend
```

## Environment Variables

```bash
# Logging
FF_LOGGING__LEVEL=INFO

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...

# OpenTelemetry (if using separate collector)
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io
OTEL_EXPORTER_OTLP_HEADERS=x-honeycomb-team=...
```
