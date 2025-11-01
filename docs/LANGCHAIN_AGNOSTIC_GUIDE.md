# MASAI LangChain Agnosticism Guide

## Overview

MASAI is **LangChain-agnostic** - it only uses LangChain for embeddings (optional). Everything else is MASAI-native.

---

## Where LangChain is Used

### Only for Embeddings (Optional)

```python
# ✅ LangChain embeddings (optional)
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = OpenAIEmbeddings()
```

### NOT Used For

- ❌ Document handling (MASAI has its own)
- ❌ Agent orchestration (MASAI uses LangGraph)
- ❌ Memory management (MASAI has its own)
- ❌ Tool integration (MASAI has its own)
- ❌ Prompt templates (MASAI has its own)

---

## Using MASAI Without LangChain

### Setup Without LangChain

```python
from masai.AgentManager import AgentManager, AgentDetails
from masai.Memory.LongTermMemory import RedisConfig

# No LangChain imports needed!

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=None  # No embeddings
)

manager = AgentManager(
    model_config_path="model_config.json",
    user_id="user_123",
    memory_config=redis_config
)

agent = manager.create_agent(
    agent_name="assistant",
    tools=[],
    agent_details=AgentDetails(
        capabilities=["reasoning"],
        description="Assistant"
    )
)

result = await agent.initiate_agent(query="Hello")
```

### Document Handling Without LangChain

```python
# ✅ MASAI Document (no LangChain)
from masai.schema import Document

doc = Document(
    page_content="Hello world",
    metadata={"source": "test"}
)

# ❌ LangChain Document (don't use)
# from langchain.schema import Document
```

---

## Custom Embedding Functions

### Simple Custom Embedder

```python
import numpy as np

def my_embedder(texts: list[str]) -> np.ndarray:
    """Simple embedding function"""
    # Your embedding logic here
    embeddings = []
    for text in texts:
        # Example: simple hash-based embedding
        embedding = np.array([hash(text) % 1000 for _ in range(1536)])
        embeddings.append(embedding)
    return np.array(embeddings)

# Use with MASAI
from masai.Memory.LongTermMemory import RedisConfig

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=my_embedder  # Custom function
)
```

### API-Based Embedder

```python
import httpx
import numpy as np

async def api_embedder(texts: list[str]) -> np.ndarray:
    """Embedding via API"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.example.com/embed",
            json={"texts": texts}
        )
        embeddings = response.json()["embeddings"]
        return np.array(embeddings)

# Use with MASAI
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=api_embedder
)
```

### Local Model Embedder

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load local model
model = SentenceTransformer("all-MiniLM-L6-v2")

def local_embedder(texts: list[str]) -> np.ndarray:
    """Local embedding model"""
    embeddings = model.encode(texts)
    return np.array(embeddings)

# Use with MASAI
redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=384,  # all-MiniLM-L6-v2 uses 384 dimensions
    embedding_model=local_embedder
)
```

---

## Alternative Embedding Providers

### OpenAI (Without LangChain)

```python
import openai
import numpy as np

async def openai_embedder(texts: list[str]) -> np.ndarray:
    """OpenAI embeddings without LangChain"""
    response = await openai.AsyncOpenAI().embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1536,
    embedding_model=openai_embedder
)
```

### Hugging Face (Without LangChain)

```python
from huggingface_hub import InferenceClient
import numpy as np

client = InferenceClient(api_key="hf_...")

def hf_embedder(texts: list[str]) -> np.ndarray:
    """Hugging Face embeddings without LangChain"""
    embeddings = []
    for text in texts:
        embedding = client.feature_extraction(text)
        embeddings.append(embedding)
    return np.array(embeddings)

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=768,
    embedding_model=hf_embedder
)
```

### Cohere (Without LangChain)

```python
import cohere
import numpy as np

co = cohere.AsyncClientV2(api_key="...")

async def cohere_embedder(texts: list[str]) -> np.ndarray:
    """Cohere embeddings without LangChain"""
    response = await co.embed(
        model="embed-english-v3.0",
        texts=texts,
        input_type="search_document"
    )
    embeddings = [item.embedding for item in response.embeddings]
    return np.array(embeddings)

redis_config = RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="masai_vectors",
    vector_size=1024,
    embedding_model=cohere_embedder
)
```

---

## Tool Integration Without LangChain

### MASAI Tool Class

```python
from masai.Tools import Tool

class SearchTool(Tool):
    name = "search"
    description = "Search the web"
    
    def execute(self, query: str) -> str:
        """Execute search"""
        # Your implementation
        return f"Results for {query}"

class CalculatorTool(Tool):
    name = "calculator"
    description = "Perform calculations"
    
    def execute(self, expression: str) -> str:
        """Execute calculation"""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

# Use with agent
tools = [SearchTool(), CalculatorTool()]

agent = manager.create_agent(
    agent_name="assistant",
    tools=tools,
    agent_details=...
)
```

### No LangChain Tool Wrapper

```python
# ✅ MASAI tools (no LangChain)
from masai.Tools import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "My custom tool"
    
    def execute(self, input: str) -> str:
        return f"Processed: {input}"

# ❌ LangChain tools (don't use)
# from langchain.tools import Tool
```

---

## Prompt Templates Without LangChain

### MASAI Prompt Templates

```python
from masai.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate

# Create prompt template
template = "You are a helpful assistant. Answer: {question}"

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# Use with agent
agent.set_context(context=prompt)
```

### No LangChain Prompts

```python
# ✅ MASAI prompts (no LangChain)
from masai.prompts import ChatPromptTemplate

# ❌ LangChain prompts (don't use)
# from langchain.prompts import ChatPromptTemplate
```

---

## Migration from LangChain-Dependent Code

### Before (LangChain-Heavy)

```python
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "My tool"
    
    def _run(self, input: str) -> str:
        return f"Result: {input}"

doc = Document(page_content="...", metadata={})
embeddings = OpenAIEmbeddings()
```

### After (MASAI-Native)

```python
from masai.schema import Document
from masai.Tools import Tool
from langchain_openai import OpenAIEmbeddings  # Only for embeddings

class MyTool(Tool):
    name = "my_tool"
    description = "My tool"
    
    def execute(self, input: str) -> str:
        return f"Result: {input}"

doc = Document(page_content="...", metadata={})
embeddings = OpenAIEmbeddings()  # Optional
```

---

## Best Practices for Agnosticism

### 1. Use MASAI Classes

```python
# ✅ GOOD - Use MASAI classes
from masai.schema import Document
from masai.Tools import Tool
from masai.prompts import ChatPromptTemplate

# ❌ BAD - Use LangChain classes
# from langchain.schema import Document
# from langchain.tools import Tool
# from langchain.prompts import ChatPromptTemplate
```

### 2. Custom Embeddings

```python
# ✅ GOOD - Custom embeddings
def my_embedder(texts):
    return embeddings

redis_config = RedisConfig(
    embedding_model=my_embedder
)

# ❌ BAD - Always use LangChain
# from langchain_openai import OpenAIEmbeddings
```

### 3. Minimal Dependencies

```python
# ✅ GOOD - Minimal imports
from masai.AgentManager import AgentManager
from masai.schema import Document

# ❌ BAD - Heavy imports
from langchain import ...
from langchain_openai import ...
from langchain_community import ...
```

---

## Troubleshooting

### Issue: "Cannot import Document from langchain"

```python
# ✅ CORRECT
from masai.schema import Document

# ❌ WRONG
from langchain.schema import Document
```

### Issue: "LangChain dependency not installed"

**Solution**: You don't need LangChain! Use MASAI-native classes.

### Issue: "Custom embedder not working"

```python
# Make sure embedder returns numpy array
def my_embedder(texts):
    import numpy as np
    embeddings = [...]
    return np.array(embeddings)  # ✅ Must be numpy array
```

---

## Summary

| Aspect | MASAI | LangChain |
|--------|-------|-----------|
| Document | ✅ Native | ❌ Don't use |
| Tools | ✅ Native | ❌ Don't use |
| Prompts | ✅ Native | ❌ Don't use |
| Embeddings | ✅ Optional | ✅ Optional |
| Agents | ✅ Native | ❌ Don't use |
| Memory | ✅ Native | ❌ Don't use |

---

## See Also

- [FRAMEWORK_OVERVIEW.md](FRAMEWORK_OVERVIEW.md) - Architecture overview
- [DOCUMENT_CLASS_GUIDE.md](DOCUMENT_CLASS_GUIDE.md) - Document class
- [REDIS_QDRANT_CRUD_GUIDE.md](REDIS_QDRANT_CRUD_GUIDE.md) - Memory operations

