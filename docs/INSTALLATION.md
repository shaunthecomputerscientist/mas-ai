# Installation & Setup

Complete guide to install and configure MASAI.

## System Requirements

- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows
- **Memory**: 4GB minimum (8GB recommended)
- **Disk**: 2GB for dependencies

## Step 1: Install MASAI

### From PyPI (Recommended)

```bash
pip install masai-framework
```

### From Source

```bash
git clone https://github.com/shaunthecomputerscientist/mas-ai.git
cd mas-ai
pip install -e .
```

### Verify Installation

```bash
python -c "import masai; print(masai.__version__)"
```

## Step 2: Install Dependencies

### Core Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# For Redis support
pip install redis

# For Qdrant support
pip install qdrant-client

# For specific LLM providers
pip install openai google-generativeai anthropic
```

## Step 3: Set Up Environment Variables

### Create .env File

```bash
cp .env.example .env
```

### Configure API Keys

Edit `.env` with your credentials:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Google Gemini
GOOGLE_API_KEY=...

# Anthropic Claude
ANTHROPIC_API_KEY=...

# Redis
REDIS_URL=redis://localhost:6379

# Qdrant
QDRANT_URL=http://localhost:6333
```

### Load Environment Variables

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

## Step 4: Set Up Memory Backend

### Option A: Redis (Recommended for Development)

#### Install Redis

**macOS**:
```bash
brew install redis
brew services start redis
```

**Linux**:
```bash
sudo apt-get install redis-server
sudo systemctl start redis-server
```

**Windows**:
```bash
# Using WSL2
wsl
sudo apt-get install redis-server
sudo service redis-server start

# Or Docker
docker run -d -p 6379:6379 redis:latest
```

#### Verify Redis

```bash
redis-cli ping
# Should return: PONG
```

### Option B: Qdrant (Recommended for Production)

#### Install Qdrant

**Docker** (Recommended):
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

**Local Installation**:
```bash
# Download from https://github.com/qdrant/qdrant/releases
./qdrant
```

#### Verify Qdrant

```bash
curl http://localhost:6333/health
# Should return: {"status":"ok"}
```

## Step 5: Verify Setup

### Test Basic Import

```python
from masai.AgentManager import AgentManager
from masai.Memory.LongTermMemory import RedisConfig

print("✓ MASAI imported successfully")
```

### Test Redis Connection

```python
import redis

r = redis.Redis(host='localhost', port=6379)
print(r.ping())  # Should print: True
```

### Test Qdrant Connection

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
print(client.get_collections())
```

### Test LLM Connection

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Hello!")
print(response.content)
```

## Step 6: Run First Example

### Create test_setup.py

```python
import asyncio
from masai.AgentManager import AgentManager, AgentDetails

async def main():
    # Create manager
    manager = AgentManager(user_id="test_user")
    
    # Create agent
    agent = manager.create_agent(
        agent_name="test_agent",
        agent_details=AgentDetails(
            capabilities=["testing"],
            description="Test agent",
            style="concise"
        )
    )
    
    # Generate response
    response = await agent.generate_response_mas(
        prompt="Say hello!",
        output_structure=None
    )
    
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Run Test

```bash
python test_setup.py
```

## Troubleshooting Installation

### Issue: "ModuleNotFoundError: No module named 'masai'"

**Solution**: Reinstall MASAI
```bash
pip uninstall masai-framework
pip install masai-framework
```

### Issue: "Redis connection refused"

**Solution**: Start Redis
```bash
redis-server
# or
docker run -d -p 6379:6379 redis:latest
```

### Issue: "OpenAI API key not found"

**Solution**: Set environment variable
```bash
export OPENAI_API_KEY="sk-..."
```

### Issue: "Qdrant connection refused"

**Solution**: Start Qdrant
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### Issue: "LangChain import error"

**Solution**: Install LangChain
```bash
pip install langchain langchain-core langchain-openai
```

## Docker Setup (Optional)

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  redis_data:
  qdrant_data:
```

### Start Services

```bash
docker-compose up -d
```

## Next Steps

1. [Quick Start Guide](QUICK_START.md)
2. [Usage Guide](USAGE_GUIDE.md)
3. [Configuration](CONFIGURATION.md)
4. [Memory System](MEMORY_SYSTEM.md)

## Getting Help

- 📖 [Documentation](../docs/)
- 🐛 [Issues](https://github.com/shaunthecomputerscientist/mas-ai/issues)
- 💬 [Discussions](https://github.com/shaunthecomputerscientist/mas-ai/discussions)

