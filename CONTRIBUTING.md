# Contributing to Multi-Agent Medical Assistant

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ChengGuoChengGou/Multi-Agent-Medical-Assistant.git
   cd Multi-Agent-Medical-Assistant
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov ruff
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Project Structure

```
├── agents/          # Multi-agent implementations (conversation, report, search, etc.)
├── api/             # External API integrations
├── config/          # Configuration and environment management
├── services/        # Core services (vector memory, semantic cache, etc.)
├── tools/           # Utility tools and helpers
├── tests/           # Unit and integration tests (1291+ tests)
├── app.py           # FastAPI application entry point
├── main.py          # Application startup
└── Makefile         # Common development commands
```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Write Code + Tests
- Follow PEP 8 style (enforced by Ruff)
- Write tests for all new functionality
- Keep test coverage above 75%

### 3. Run Tests
```bash
# All tests
make test

# With coverage
make coverage

# Specific test file
pytest tests/test_your_module.py -v
```

### 4. Lint and Format
```bash
make lint
make format
```

### 5. Commit and Push
```bash
git add .
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
```

### 6. Open a Pull Request
- Target the `dev` branch
- Include a clear description of changes
- Reference any related issues

## Code Style

- **Python 3.11+** required
- **Ruff** for linting and formatting (config in `pyproject.toml`)
- Maximum line length: 120 characters (exceptions for E501)
- Use type hints where practical

## Testing Guidelines

- **Unit tests**: Test individual functions in isolation (mock external dependencies)
- **Integration tests**: Test API endpoints using FastAPI TestClient
- **Naming**: `test_<module>_<scenario>.py`
- **Fixtures**: Use shared fixtures in `tests/conftest.py`
- **Mocking**: Mock heavy dependencies (LLM calls, vector DB, external APIs)

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Description |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `test:` | Adding or updating tests |
| `docs:` | Documentation changes |
| `refactor:` | Code refactoring (no behavior change) |
| `ci:` | CI/CD pipeline changes |
| `chore:` | Maintenance tasks |

## Agent Development

When adding a new agent:

1. Create `agents/your_agent.py` with a class inheriting from `BaseAgent`
2. Register in `config/agent_registry.py`
3. Add to `ENABLED_AGENTS` in `.env`
4. Write unit tests in `tests/test_your_agent.py`
5. Update README.md with agent description

## Questions?

Open an issue on GitHub or reach out to the maintainers.
