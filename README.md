# Foundry Local AI Agent 🚀

A privacy-first, modular AI Agent architecture designed for the Microsoft Foundry Local ecosystem, featuring local RAG (Retrieval-Augmented Generation) and strict Pydantic data validation.

---

## 📌 Features

- **Privacy-First & On-Premise:** Processes queries locally without leaking corporate data or logs to external servers.
- **Lightweight RAG Engine:** Dynamically indexes and retrieves contextual knowledge from local enterprise documents.
- **Type-Safe OOP Architecture:** Strict data contracts and response schemas powered by **Pydantic**.
- **Interactive CLI & Fallback Handling:** Seamless user interactions with error recovery mechanisms.

---

## 🛠️ Project Structure

```text
Foundry-Local-AI-Agent/
│
├── data/
│   └── knowledge.txt      # Enterprise knowledge base (Project, HR, Tech)
│
├── src/
│   ├── agent.py           # Core agent logic and orchestration
│   ├── rag.py             # Document indexing and retrieval engine
│   └── schemas.py         # Pydantic data schemas
│
├── app.py                 # Interactive terminal runner
├── requirements.txt       # Project dependencies
└── README.md              # Documentation