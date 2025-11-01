# Advanced Topics

Expert-level usage patterns and customization.

## Custom Tool Development

### Create Complex Tools

```python
from langchain.tools import tool
from typing import Optional

@tool
def advanced_search(
    query: str,
    filters: Optional[dict] = None,
    limit: int = 10
) -> str:
    """Advanced search with filters"""
    # Implement search logic
    results = []
    if filters:
        # Apply filters
        pass
    return str(results[:limit])

@tool
def data_analysis(
    data: list[float],
    operation: str = "mean"
) -> float:
    """Analyze numerical data"""
    if operation == "mean":
        return sum(data) / len(data)
    elif operation == "median":
        sorted_data = sorted(data)
        return sorted_data[len(data)//2]
    elif operation == "std":
        mean = sum(data) / len(data)
        variance = sum((x - mean)**2 for x in data) / len(data)
        return variance ** 0.5
```

### Tool with State

```python
class DatabaseTool:
    def __init__(self, connection_string):
        self.connection = self._connect(connection_string)
    
    def _connect(self, conn_str):
        # Connect to database
        pass
    
    @tool
    def query_database(self, sql: str) -> str:
        """Execute SQL query"""
        results = self.connection.execute(sql)
        return str(results)
    
    @tool
    def insert_data(self, table: str, data: dict) -> str:
        """Insert data into table"""
        # Insert logic
        return "Data inserted"

# Use in agent
db_tool = DatabaseTool("connection_string")
agent = manager.create_agent(
    ...,
    tools=[db_tool.query_database, db_tool.insert_data]
)
```

## Custom Embeddings

### Implement Custom Embedding Model

```python
from langchain_core.embeddings import Embeddings
import numpy as np

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Load your model
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents"""
        embeddings = []
        for text in texts:
            embedding = self._embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query"""
        return self._embed_text(text)
    
    def _embed_text(self, text: str) -> list[float]:
        # Your embedding logic
        return np.random.rand(1536).tolist()

# Use in agent
embeddings = CustomEmbeddings("my-model")
redis_config = RedisConfig(
    ...,
    embedding_model=embeddings
)
```

## Custom Memory Backends

### Implement Custom Backend

```python
from masai.Memory.LongTermMemory import BaseAdapter
from langchain_core.documents import Document
from typing import List, Optional

class CustomAdapter(BaseAdapter):
    def __init__(self, config):
        self.config = config
        self._initialize()
    
    def _initialize(self):
        # Initialize your backend
        pass
    
    async def save(
        self,
        user_id: str,
        documents: List[Document],
        embed_fn
    ) -> None:
        """Save documents"""
        for doc in documents:
            embedding = embed_fn.embed_query(doc.page_content)
            # Store in your backend
            self._store(user_id, doc, embedding)
    
    async def search(
        self,
        user_id: str,
        query: str,
        k: int,
        embed_fn,
        categories: Optional[List[str]] = None
    ) -> List[Document]:
        """Search documents"""
        query_embedding = embed_fn.embed_query(query)
        # Search in your backend
        results = self._search(user_id, query_embedding, k, categories)
        return results
    
    def _store(self, user_id, doc, embedding):
        # Your storage logic
        pass
    
    def _search(self, user_id, embedding, k, categories):
        # Your search logic
        pass
```

## Custom Agent Architectures

### Extend Agent Class

```python
from masai.Agent import Agent

class CustomAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_state = {}
    
    async def generate_response_mas(self, prompt, output_structure=None, **kwargs):
        # Custom preprocessing
        processed_prompt = self._preprocess(prompt)
        
        # Call parent method
        response = await super().generate_response_mas(
            processed_prompt,
            output_structure,
            **kwargs
        )
        
        # Custom postprocessing
        processed_response = self._postprocess(response)
        
        return processed_response
    
    def _preprocess(self, prompt):
        # Your preprocessing logic
        return prompt
    
    def _postprocess(self, response):
        # Your postprocessing logic
        return response
```

## Multi-Agent Workflows

### Sequential Workflow

```python
async def sequential_workflow(query):
    # Step 1: Research
    research_result = await researcher.generate_response_mas(
        prompt=f"Research: {query}",
        output_structure=None
    )
    
    # Step 2: Write
    write_result = await writer.generate_response_mas(
        prompt=f"Write based on: {research_result}",
        output_structure=None
    )
    
    # Step 3: Edit
    final_result = await editor.generate_response_mas(
        prompt=f"Edit: {write_result}",
        output_structure=None
    )
    
    return final_result
```

### Parallel Workflow

```python
import asyncio

async def parallel_workflow(query):
    # Run multiple agents in parallel
    results = await asyncio.gather(
        researcher.generate_response_mas(prompt=f"Research: {query}"),
        analyzer.generate_response_mas(prompt=f"Analyze: {query}"),
        summarizer.generate_response_mas(prompt=f"Summarize: {query}")
    )
    
    # Combine results
    combined = "\n".join(results)
    
    # Final synthesis
    final = await synthesizer.generate_response_mas(
        prompt=f"Synthesize: {combined}",
        output_structure=None
    )
    
    return final
```

### Conditional Workflow

```python
async def conditional_workflow(query):
    # Initial analysis
    analysis = await analyzer.generate_response_mas(
        prompt=f"Analyze: {query}",
        output_structure=None
    )
    
    # Conditional routing
    if "complex" in analysis.lower():
        result = await expert.generate_response_mas(
            prompt=f"Expert analysis: {query}",
            output_structure=None
        )
    else:
        result = await assistant.generate_response_mas(
            prompt=query,
            output_structure=None
        )
    
    return result
```

## Performance Tuning

### Batch Processing

```python
async def batch_process(queries: list[str], batch_size: int = 5):
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        
        # Process batch in parallel
        batch_results = await asyncio.gather(*[
            agent.generate_response_mas(prompt=q)
            for q in batch
        ])
        
        results.extend(batch_results)
    
    return results
```

### Caching Responses

```python
from functools import lru_cache

class CachedAgent:
    def __init__(self, agent):
        self.agent = agent
        self.cache = {}
    
    async def generate_response_mas(self, prompt, **kwargs):
        cache_key = hash(prompt)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        response = await self.agent.generate_response_mas(
            prompt,
            **kwargs
        )
        
        self.cache[cache_key] = response
        return response
```

## Monitoring and Logging

### Custom Logging

```python
import logging

class MASAILogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler("masai.log")
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_query(self, query):
        self.logger.info(f"Query: {query}")
    
    def log_response(self, response):
        self.logger.info(f"Response: {response}")
    
    def log_error(self, error):
        self.logger.error(f"Error: {error}")
```

### Performance Metrics

```python
import time

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    async def measure(self, name, coro):
        start = time.time()
        result = await coro
        duration = time.time() - start
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(duration)
        return result
    
    def get_stats(self, name):
        times = self.metrics.get(name, [])
        if not times:
            return None
        
        return {
            "count": len(times),
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times)
        }
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

