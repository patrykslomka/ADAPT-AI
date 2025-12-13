# ADAPT-AI: Clinical Decision Support System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A configurable multi-agent architecture for AI systems using Model Context Protocol (MCP) and modular building blocks, demonstrated in healthcare clinical diagnostics.

**🎓 Academic Context:** This implementation validates the ADAPT-AI framework proposed in a Master's thesis on modular AI architectures.

---

## 🌟 Key Features

- **🤖 3 Specialized AI Agents**
  - Primary Clinical Agent (diagnostic reasoning)
  - Compliance Agent (HIPAA/FDA validation)
  - Quality Agent (hallucination detection)

- **🎛️ MCP Orchestration**
  - Centralized agent coordination
  - Feedback loops for quality assurance
  - Session & context management

- **🧠 Dual Reasoning Systems**
  - RAG: Fast retrieval for simple queries
  - RAT: Multi-step reasoning for complex diagnostics

- **📊 Complete LLMOps Stack**
  - Real-time metrics collection
  - Distributed tracing
  - Automated alerting
  - Interactive dashboard

- **🏥 Healthcare-Ready**
  - 20 synthetic patient records
  - Clinical ontology (diseases, symptoms, treatments)
  - Drug interaction database
  - HIPAA compliance checks

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Streamlit Dashboard             │
├─────────────────────────────────────────┤
│          MCP Orchestrator               │
│  ┌──────┐  ┌──────────┐  ┌────────┐   │
│  │Primary│  │Compliance│  │Quality │   │
│  │ Agent │──│  Agent   │──│ Agent  │   │
│  └───┬───┘  └──────────┘  └────────┘   │
│      │                                   │
│  ┌───▼──────┐        ┌──────────────┐  │
│  │RAG / RAT │        │  Validation  │  │
│  └──────────┘        └──────────────┘  │
├─────────────────────────────────────────┤
│      Domain Configuration               │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │Ontologies│ │ Patients │ │  HIPAA │ │
│  └──────────┘ └──────────┘ └────────┘ │
└─────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Installation

```bash
# 1. Navigate to project directory
cd ADAPT-AI

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env
# Edit .env with your Anthropic API key

# 6. Generate synthetic data
python scripts/generate_patients.py

# 7. Seed vector database
python scripts/seed_vector_db.py

# 8. Run tests
pytest tests/ -v
```

### First Query (Day 2)

After completing Day 2 implementation:

```bash
# Launch application
streamlit run ui/app.py
```

1. Open http://localhost:8501
2. Select a patient (e.g., P-0001)
3. Ask: *"Suggest diagnostic workup for this patient's presenting complaint"*
4. Watch agents collaborate in real-time!

---

## 📁 Project Structure

```
ADAPT-AI/
├── config/                 # Configuration management
│   └── settings.py         # Pydantic settings
├── src/
│   ├── agents/             # AI agents (Day 2)
│   ├── building_blocks/    # RAG & RAT modules
│   │   ├── rag.py          # Retrieval-Augmented Generation
│   │   └── rat.py          # Retrieval-Augmented Thoughts
│   ├── domain/
│   │   ├── compliance/     # HIPAA/FDA rules
│   │   ├── ontologies/     # Clinical knowledge
│   │   └── synthetic_patients/
│   ├── llmops/             # Metrics & monitoring
│   ├── mcp/                # Orchestrator (Day 2)
│   └── utils/              # Logger, helpers
├── tests/                  # Test suites
├── ui/                     # Streamlit UI (Day 3)
├── scripts/                # Setup & generation scripts
├── data/                   # Runtime data (gitignored)
└── docs/                   # Documentation
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_domain/ -v          # Domain tests
pytest tests/test_building_blocks/ -v # RAG/RAT tests

# Generate coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 🔒 Security & Compliance

- **No hardcoded secrets** - All sensitive data in `.env`
- **HIPAA-compliant patterns** - PHI protection built-in
- **Input validation** - Pydantic models throughout
- **Structured logging** - Sensitive data redaction
- **Rate limiting** - API abuse prevention

---

## 📊 Day 1 Deliverables

| Component | Status |
|-----------|--------|
| Project Structure | ✅ |
| Settings Management | ✅ |
| Secure Logger | ✅ |
| Clinical Ontology | ✅ |
| Drug Database | ✅ |
| HIPAA/FDA Rules | ✅ |
| Ontology Loader | ✅ |
| Synthetic Patients | ✅ |
| RAG Building Block | ✅ |
| RAT Building Block | ✅ |
| LLMOps Metrics | ✅ |

---

## 🗓️ Implementation Timeline

- **Day 1**: Foundation (domain config, RAG/RAT, metrics) ✅
- **Day 2**: Agents & Integration (MCP, 3 agents, testing)
- **Day 3**: UI & Polish (Streamlit, documentation, demo)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

**⭐ Star this repo if you find it helpful!**
