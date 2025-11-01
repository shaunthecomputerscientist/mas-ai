# OMAN (Orchestrated Multi-Agent Network) Guide

Complete guide for using MASAI's Orchestrated Multi-Agent Network (OMAN) to coordinate multiple Multi-Agent Systems.

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is OMAN?](#what-is-oman)
3. [Architecture](#architecture)
4. [Setup and Configuration](#setup-and-configuration)
5. [Usage](#usage)
6. [Data Flow](#data-flow)
7. [Complete Examples](#complete-examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

OMAN (Orchestrated Multi-Agent Network) is MASAI's highest level of orchestration, enabling coordination between multiple Multi-Agent Systems (MAS). It's designed for large-scale, multi-domain applications where different specialized MAS instances need to work together.

### When to Use OMAN

Use OMAN when:
- ✅ You have multiple specialized domains (e.g., Finance, Research, Support)
- ✅ Each domain requires its own set of specialized agents
- ✅ You need intelligent routing between domains
- ✅ You want to scale your system by adding new MAS instances
- ✅ You need enterprise-level multi-agent orchestration

Don't use OMAN when:
- ❌ Single MAS is sufficient for your use case
- ❌ All agents can work within one domain
- ❌ Simple agent coordination is enough

---

## What is OMAN?

OMAN is a **network-level orchestrator** that:
- Manages multiple MAS instances
- Routes queries to the appropriate MAS based on capabilities
- Maintains network-wide shared memory
- Coordinates cross-domain tasks
- Scales horizontally by adding new MAS instances

### OMAN vs MAS

| Feature | MAS | OMAN |
|---------|-----|------|
| **Scope** | Single domain | Multiple domains |
| **Agents** | Multiple agents | Multiple MAS instances |
| **Routing** | Agent-to-agent | MAS-to-MAS |
| **Use Case** | Domain-specific tasks | Enterprise-wide tasks |
| **Complexity** | Medium | High |

---

## Architecture

### OMAN Components

```
┌─────────────────────────────────────────────┐
│           OMAN Supervisor (LLM)             │
│  - Analyzes queries                         │
│  - Routes to appropriate MAS                │
│  - Maintains network memory                 │
└─────────────────────────────────────────────┘
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
┌─────────┐   ┌─────────┐   ┌─────────┐
│Finance  │   │Research │   │Support  │
│  MAS    │   │  MAS    │   │  MAS    │
└─────────┘   └─────────┘   └─────────┘
    ↓               ↓               ↓
┌─────────┐   ┌─────────┐   ┌─────────┐
│Agents   │   │Agents   │   │Agents   │
│(3-5)    │   │(3-5)    │   │(3-5)    │
└─────────┘   └─────────┘   └─────────┘
```

### Key Components

1. **OMAN Supervisor**: LLM-based router that analyzes queries and selects appropriate MAS
2. **MAS Instances**: Specialized Multi-Agent Systems for different domains
3. **Shared Memory**: Network-wide memory tracking tasks, outcomes, and capabilities
4. **Task Queue**: Manages concurrent task execution across networks

---

## Setup and Configuration

### Basic Setup

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.MultiAgents.MultiAgent import MultiAgentSystem
from masai.OMAN.oman import OrchestratedMultiAgentNetwork

# Step 1: Create specialized AgentManagers for each domain
finance_manager = AgentManager(
    user_id="finance_team",
    model_config_path="model_config.json"
)

research_manager = AgentManager(
    user_id="research_team",
    model_config_path="model_config.json"
)

support_manager = AgentManager(
    user_id="support_team",
    model_config_path="model_config.json"
)

# Step 2: Create agents for each domain
# Finance agents
finance_agent1 = finance_manager.create_agent(
    agent_name="financial_analyst",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["financial analysis", "revenue tracking", "budget planning"],
        description="Analyzes financial data and provides insights",
        style="precise and data-driven"
    )
)

finance_agent2 = finance_manager.create_agent(
    agent_name="accountant",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["accounting", "tax calculation", "expense tracking"],
        description="Handles accounting and tax-related tasks",
        style="meticulous and compliant"
    )
)

# Research agents
research_agent1 = research_manager.create_agent(
    agent_name="researcher",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["research", "data gathering", "analysis"],
        description="Conducts research and gathers information",
        style="thorough and analytical"
    )
)

research_agent2 = research_manager.create_agent(
    agent_name="data_analyst",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["data analysis", "visualization", "reporting"],
        description="Analyzes data and creates reports",
        style="clear and visual"
    )
)

# Support agents
support_agent1 = support_manager.create_agent(
    agent_name="support_specialist",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["customer support", "troubleshooting", "issue resolution"],
        description="Provides customer support and resolves issues",
        style="empathetic and helpful"
    )
)

# Step 3: Create MAS instances for each domain
finance_mas = MultiAgentSystem(agentManager=finance_manager)
research_mas = MultiAgentSystem(agentManager=research_manager)
support_mas = MultiAgentSystem(agentManager=support_manager)

# Step 4: Create OMAN
oman = OrchestratedMultiAgentNetwork(
    mas_instances=[finance_mas, research_mas, support_mas],
    network_memory_order=5,
    oman_llm_config={
        "model_name": "gpt-4o",
        "category": "openai",
        "temperature": 0.2,
        "memory_order": 5
    },
    extra_context={
        "environment": "production",
        "company": "MyCompany"
    }
)
```

### Configuration Parameters

#### `mas_instances`
- **Type**: `List[MultiAgentSystem]`
- **Description**: List of MAS instances to orchestrate
- **Required**: Yes

#### `network_memory_order`
- **Type**: `int`
- **Default**: `3`
- **Description**: Number of past tasks to keep in network memory

#### `oman_llm_config`
- **Type**: `Dict`
- **Description**: Configuration for OMAN supervisor LLM
- **Structure**:
  ```python
  {
      "model_name": "gpt-4o",      # Model for routing
      "category": "openai",         # Provider category
      "temperature": 0.2,           # Sampling temperature
      "memory_order": 5             # Context window size
  }
  ```

#### `extra_context`
- **Type**: `Dict`
- **Description**: Additional context for OMAN supervisor
- **Example**: `{"environment": "production", "company": "MyCompany"}`

---

## Usage

### Delegating Tasks

```python
# Delegate a query to OMAN
result = oman.delegate_task(
    query="What is our Q3 revenue and how does it compare to market trends?"
)

print(result)
```

### How OMAN Routes Queries

1. **Query Analysis**: OMAN supervisor analyzes the query
2. **Capability Matching**: Compares query against each MAS's agent capabilities
3. **MAS Selection**: Selects the most appropriate MAS
4. **Task Delegation**: Delegates query to selected MAS
5. **Execution**: MAS processes query using its agents
6. **Result Return**: Result returned through OMAN supervisor
7. **Memory Update**: Network memory updated with task outcome

### Accessing Shared Memory

```python
# Access OMAN shared memory
shared_memory = oman.shared_memory

print(f"Completed tasks: {shared_memory['tasks']}")
print(f"Task outcomes: {shared_memory['outcomes']}")
print(f"Network capabilities: {shared_memory['capabilities']}")
```

---

## Data Flow

### Query Routing Flow

```
User Query
    ↓
OMAN Supervisor (analyzes query)
    ↓
Capability Matching
    ├─ Finance MAS: ["financial analysis", "revenue tracking", "budget planning", "accounting"]
    ├─ Research MAS: ["research", "data gathering", "analysis", "visualization"]
    └─ Support MAS: ["customer support", "troubleshooting", "issue resolution"]
    ↓
Selected MAS (e.g., Finance MAS)
    ↓
MAS Agents (financial_analyst, accountant)
    ↓
Agent Processing
    ↓
Result
    ↓
OMAN Supervisor (aggregates result)
    ↓
Final Answer
```

### Memory Flow

```
Task Execution
    ↓
Task Outcome
    ↓
OMAN Shared Memory
    ├─ tasks: [task1, task2, ...]
    ├─ outcomes: [outcome1, outcome2, ...]
    └─ capabilities: {network1: [...], network2: [...]}
    ↓
Future Query Context
```

---

## Complete Examples

### Example 1: Enterprise System

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.MultiAgents.MultiAgent import MultiAgentSystem
from masai.OMAN.oman import OrchestratedMultiAgentNetwork

# Create domain-specific managers
finance_manager = AgentManager(
    user_id="finance",
    model_config_path="model_config.json"
)

hr_manager = AgentManager(
    user_id="hr",
    model_config_path="model_config.json"
)

it_manager = AgentManager(
    user_id="it",
    model_config_path="model_config.json"
)

# Create agents for each domain
# Finance
finance_analyst = finance_manager.create_agent(
    agent_name="analyst",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["financial analysis", "reporting"],
        description="Financial analyst"
    )
)

# HR
hr_specialist = hr_manager.create_agent(
    agent_name="hr_specialist",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["recruitment", "employee management"],
        description="HR specialist"
    )
)

# IT
it_support = it_manager.create_agent(
    agent_name="it_support",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["technical support", "system maintenance"],
        description="IT support specialist"
    )
)

# Create MAS instances
finance_mas = MultiAgentSystem(agentManager=finance_manager)
hr_mas = MultiAgentSystem(agentManager=hr_manager)
it_mas = MultiAgentSystem(agentManager=it_manager)

# Create OMAN
oman = OrchestratedMultiAgentNetwork(
    mas_instances=[finance_mas, hr_mas, it_mas],
    network_memory_order=5,
    oman_llm_config={
        "model_name": "gpt-4o",
        "category": "openai",
        "temperature": 0.2,
        "memory_order": 5
    },
    extra_context={"company": "TechCorp"}
)

# Use OMAN
queries = [
    "What is our Q4 budget status?",  # → Finance MAS
    "How many open positions do we have?",  # → HR MAS
    "Is the server maintenance scheduled?",  # → IT MAS
]

for query in queries:
    result = oman.delegate_task(query)
    print(f"Query: {query}")
    print(f"Answer: {result}\n")
```

### Example 2: Multi-Domain Research Platform

```python
# Create specialized research domains
academic_manager = AgentManager(user_id="academic", model_config_path="model_config.json")
market_manager = AgentManager(user_id="market", model_config_path="model_config.json")
tech_manager = AgentManager(user_id="tech", model_config_path="model_config.json")

# Academic research agents
academic_researcher = academic_manager.create_agent(
    agent_name="academic_researcher",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["academic research", "literature review", "citation analysis"],
        description="Conducts academic research and literature reviews"
    )
)

# Market research agents
market_analyst = market_manager.create_agent(
    agent_name="market_analyst",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["market analysis", "competitor research", "trend analysis"],
        description="Analyzes market trends and competitors"
    )
)

# Tech research agents
tech_researcher = tech_manager.create_agent(
    agent_name="tech_researcher",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["technology research", "product analysis", "innovation tracking"],
        description="Researches technology and innovation"
    )
)

# Create MAS instances
academic_mas = MultiAgentSystem(agentManager=academic_manager)
market_mas = MultiAgentSystem(agentManager=market_manager)
tech_mas = MultiAgentSystem(agentManager=tech_manager)

# Create OMAN
research_oman = OrchestratedMultiAgentNetwork(
    mas_instances=[academic_mas, market_mas, tech_mas],
    network_memory_order=10,
    oman_llm_config={
        "model_name": "gemini-2.5-pro",
        "category": "gemini",
        "temperature": 0.3,
        "memory_order": 10
    },
    extra_context={"platform": "ResearchHub"}
)

# Complex multi-domain query
result = research_oman.delegate_task(
    "What are the latest academic papers on AI, and how do they relate to current market trends?"
)
print(result)
```

---

## Best Practices

### 1. Domain Specialization

**✅ DO**: Create highly specialized MAS instances
```python
# Good: Clear domain separation
finance_mas = MultiAgentSystem(agentManager=finance_manager)  # Only finance agents
research_mas = MultiAgentSystem(agentManager=research_manager)  # Only research agents
```

**❌ DON'T**: Mix unrelated capabilities in one MAS
```python
# Bad: Mixed capabilities
general_mas = MultiAgentSystem(agentManager=general_manager)  # Finance + Research + Support
```

### 2. Agent Capabilities

**✅ DO**: Define clear, specific capabilities
```python
agent_details=AgentDetails(
    capabilities=["financial analysis", "revenue tracking", "budget planning"],
    description="Analyzes financial data and provides insights"
)
```

**❌ DON'T**: Use vague or overlapping capabilities
```python
agent_details=AgentDetails(
    capabilities=["general tasks", "anything"],
    description="Does stuff"
)
```

### 3. Memory Management

**✅ DO**: Set appropriate memory order based on task complexity
```python
# Simple tasks
oman = OrchestratedMultiAgentNetwork(
    mas_instances=[...],
    network_memory_order=3  # Keep last 3 tasks
)

# Complex tasks requiring more context
oman = OrchestratedMultiAgentNetwork(
    mas_instances=[...],
    network_memory_order=10  # Keep last 10 tasks
)
```

### 4. LLM Configuration

**✅ DO**: Use powerful models for OMAN supervisor
```python
oman_llm_config={
    "model_name": "gpt-4o",  # Strong reasoning for routing
    "category": "openai",
    "temperature": 0.2  # Low temperature for consistent routing
}
```

**❌ DON'T**: Use weak models for routing
```python
oman_llm_config={
    "model_name": "gpt-3.5-turbo",  # May struggle with complex routing
    "temperature": 0.9  # High temperature causes inconsistent routing
}
```

### 5. Scaling Strategy

**✅ DO**: Add new MAS instances as needed
```python
# Start with 3 domains
oman = OrchestratedMultiAgentNetwork(
    mas_instances=[finance_mas, research_mas, support_mas],
    ...
)

# Later, add new domain
legal_mas = MultiAgentSystem(agentManager=legal_manager)
oman.networks["Legal"] = legal_mas  # Add new network
```

### 6. Error Handling

**✅ DO**: Handle routing errors gracefully
```python
try:
    result = oman.delegate_task(query)
    print(result)
except ValueError as e:
    print(f"Routing error: {e}")
except Exception as e:
    print(f"Execution error: {e}")
```

---

## Troubleshooting

### Issue 1: Wrong MAS Selected

**Problem**: OMAN routes query to wrong MAS

**Causes**:
- Overlapping agent capabilities
- Vague capability descriptions
- Insufficient context in query

**Solutions**:
```python
# 1. Make capabilities more specific
agent_details=AgentDetails(
    capabilities=["financial_analysis", "revenue_tracking"],  # Specific
    description="Analyzes financial data and tracks revenue"
)

# 2. Provide more context in query
result = oman.delegate_task(
    "Analyze Q3 financial performance including revenue and expenses"  # Clear context
)

# 3. Use extra_context to guide routing
oman = OrchestratedMultiAgentNetwork(
    mas_instances=[...],
    extra_context={
        "finance_keywords": ["revenue", "budget", "expenses"],
        "research_keywords": ["analysis", "trends", "data"]
    }
)
```

### Issue 2: Slow Routing

**Problem**: OMAN takes too long to route queries

**Causes**:
- Too many MAS instances
- Large network memory
- Slow LLM model

**Solutions**:
```python
# 1. Reduce network memory
oman = OrchestratedMultiAgentNetwork(
    mas_instances=[...],
    network_memory_order=3  # Reduce from 10 to 3
)

# 2. Use faster model for routing
oman_llm_config={
    "model_name": "gemini-2.5-flash",  # Faster than gpt-4o
    "category": "gemini"
}

# 3. Limit number of MAS instances (< 10)
```

### Issue 3: Memory Not Persisting

**Problem**: Network memory doesn't persist across sessions

**Cause**: OMAN shared memory is in-memory only

**Solution**: Implement custom persistence
```python
import json

# Save memory
with open("oman_memory.json", "w") as f:
    json.dump(oman.shared_memory, f)

# Load memory
with open("oman_memory.json", "r") as f:
    oman.shared_memory = json.load(f)
```

### Issue 4: Network Not Found Error

**Problem**: `ValueError: Routed network 'X' not found in OMAN`

**Cause**: OMAN supervisor returned invalid network name

**Solution**:
```python
# Check available networks
print(f"Available networks: {list(oman.networks.keys())}")

# Verify network names match
for name, mas in oman.networks.items():
    print(f"Network: {name}")
    print(f"Agents: {list(mas.agentManager.agents.keys())}")
```

---

## Advanced Topics

### Custom Routing Logic

You can extend OMAN with custom routing logic:

```python
class CustomOrchestratedMultiAgentNetwork(OrchestratedMultiAgentNetwork):
    def delegate_task(self, query: str) -> str:
        # Custom pre-processing
        if "urgent" in query.lower():
            # Route urgent queries to priority MAS
            selected_network = self.networks["Priority"]
        else:
            # Use default routing
            return super().delegate_task(query)

        # Execute in selected network
        result = selected_network.initiate_decentralized_mas(
            query=query,
            set_entry_agent=list(selected_network.agentManager.agents.values())[0]
        )
        return result["answer"]
```

### Monitoring and Logging

```python
# Monitor OMAN activity
print(f"Total tasks: {len(oman.shared_memory['tasks'])}")
print(f"Total outcomes: {len(oman.shared_memory['outcomes'])}")

# Log routing decisions
for task, outcome in zip(oman.shared_memory['tasks'], oman.shared_memory['outcomes']):
    print(f"Task: {task}")
    print(f"Outcome: {outcome}\n")
```

---

## Summary

OMAN provides enterprise-level orchestration for multi-domain agent systems:

| Feature | Description |
|---------|-------------|
| **Multi-Domain** | Coordinate multiple specialized MAS instances |
| **Intelligent Routing** | LLM-based query routing to appropriate MAS |
| **Scalable** | Add new MAS instances without modifying existing ones |
| **Shared Memory** | Network-wide memory for context and coordination |
| **Flexible** | Support for any number of domains and agents |

**Next Steps**:
- [MultiAgent System Guide](MULTIAGENT_SYSTEM_GUIDE.md) - Learn about MAS
- [Model Parameters](MODEL_PARAMETERS.md) - Configure OMAN supervisor LLM
- [Tools Guide](TOOLS_GUIDE.md) - Add tools to OMAN agents
- [API Reference](API_REFERENCE.md) - Complete API documentation
```

