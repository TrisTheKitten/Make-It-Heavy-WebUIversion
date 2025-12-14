# Make It Heavy - Web Edition

A web-based interface for the **Make It Heavy** multi-agent AI research framework. This project provides a browser-accessible UI for running single-agent queries and parallel multi-agent orchestration.

> **Based on [Make It Heavy](https://github.com/Doriandarko/make-it-heavy) by [Pietro Schirano](https://github.com/Doriandarko)**  
> This web version builds upon the original CLI-based multi-agent framework, adding a Flask-powered web interface with real-time streaming progress.

## Features

- **Web Interface** - Modern browser-based UI accessible at `http://localhost:5050`
- **Single Agent Mode** - Direct queries to an AI agent with tool access
- **Parallel Orchestrator** - Multiple agents working on decomposed subtasks simultaneously
- **Megamind Mode** - Advanced orchestration with enhanced task decomposition and synthesis
- **Real-time Streaming** - Server-sent events for live progress updates
- **Image Support** - Attach images to queries for multimodal analysis
- **Multi-Provider Support** - Works with both Gemini and OpenRouter APIs
- **Configurable** - Adjust models, parallel agents, timeouts, and iterations via UI or config file

## Tools

The agent comes equipped with:
- **Web Search** - DuckDuckGo search integration
- **Calculator** - Mathematical expression evaluation
- **Read File** - Read local file contents
- **Write File** - Write content to local files
- **Task Complete** - Signal task completion with summary

## Installation

```bash
git clone <repo-url>
cd make-it-heavy

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to set your API keys and preferences:

```yaml
provider: gemini  # or 'openrouter'

gemini:
  api_key: YOUR_GEMINI_API_KEY
  model: gemini-2.5-flash

openrouter:
  api_key: YOUR_OPENROUTER_API_KEY
  base_url: https://openrouter.ai/api/v1
  model: moonshotai/kimi-k2

orchestrator:
  parallel_agents: 4
  task_timeout: 300
  aggregation_strategy: consensus

agent:
  max_iterations: 12
```

## Usage

### Web Server

```bash
python web_server.py
```

Open `http://localhost:5050` in your browser.

### CLI Mode

**Single Agent:**
```bash
python main.py
```

**Multi-Agent Orchestrator:**
```bash
python make_it_heavy.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve web interface |
| `/api/config` | GET | Get current configuration |
| `/api/config` | POST | Update configuration |
| `/api/run/single` | POST | Run single agent query |
| `/api/run/orchestrator` | POST | Run parallel orchestrator |
| `/api/run/megamind` | POST | Run megamind orchestrator |
| `/api/stream/<session_id>` | GET | SSE stream for progress updates |

## Architecture

```
├── web_server.py          # Flask web server with SSE streaming
├── agent.py               # Base agent with Gemini/OpenRouter support
├── orchestrator.py        # Parallel task orchestrator
├── megamind_orchestrator.py  # Advanced orchestrator with synthesis
├── make_it_heavy.py       # CLI interface for orchestrator
├── main.py                # CLI interface for single agent
├── config.yaml            # Configuration file
├── tools/                 # Tool implementations
│   ├── search_tool.py
│   ├── calculator_tool.py
│   ├── read_file_tool.py
│   ├── write_file_tool.py
│   └── task_done_tool.py
└── static/
    └── index.html         # Web UI
```

## Credits

- **Original Framework**: [Make It Heavy](https://github.com/Doriandarko/make-it-heavy) by [Pietro Schirano](https://github.com/Doriandarko)
- **License**: MIT License with Attribution Requirement for Large-Scale Commercial Use

If using in a product serving 100,000+ users, attribution to Pietro Schirano is required. See [LICENSE](LICENSE) for details.
