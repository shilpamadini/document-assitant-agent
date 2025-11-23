# Document Assistant Agent

Welcome to the Document Assistant project! This project will help you build a sophisticated document processing system using LangChain and LangGraph. You'll create an AI assistant that can answer questions, summarize documents, and perform calculations on financial and healthcare documents.

## Project Overview

This document assistant uses a multi-agent architecture with LangGraph to handle different types of user requests:
- **Q&A Agent**: Answers specific questions about document content
- **Summarization Agent**: Creates summaries and extracts key points from documents
- **Calculation Agent**: Performs mathematical operations on document data

### Prerequisites
- Python 3.9+
- OpenAI API key
- (Optional but recommended) Conda for environment management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/shilpamadini/document-assitant-agent.git
cd document-assitant-agent/project
```

2. Create a virtual environment - Using Conda:
```bash
conda create -n doc-assistant python=3.10 -y
conda activate doc-assistant
```

3. Install dependencies:
```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name doc-assistant --display-name "Doc Assistant"
pip install -r requirements.txt
```

4. Create a `.env` file:
```bash
cp env.example .env
# Edit .env and add your OpenAI API key
```

```bash
EXPORT OPENAI_API_KEY=your_openai_api_key_here
```
Make sure .env is listed in .gitignore so your API key is never committed to the repository.

### Running the Assistant

```bash
python main.py
```
This will:
1. Load the assistant
2. Start a new session
3. Accept user input
4. Route the message through the LangGraph agent system

Displays:
1. Response
2. Intent classification
3. Tools used
4. Updated memory
5. Active document references

### Running Unit Tests

All unit tests live in the tests/ directory and use the standard unittest framework.
```
python -m unittest discover -s tests -p "test_*.py"
```

## Project Structure
```
# Document Assistant Project

This project implements an intelligent multi-agent document assistant using **Python**, **LangChain**, and **LangGraph**.  
The assistant can:

- Answer questions about financial and healthcare documents (Q&A agent)
- Summarize documents and extract key points (Summarization agent)
- Perform calculations over document data using tools (Calculation agent)

---

## Project Structure

```text
project/
├── main.py              
├── requirements.txt     
├── README.md            
├── src/
│   ├── __init__.py      
│   ├── agent.py         # LangGraph workflow and AgentState
│   ├── assistant.py     # DocumentAssistant class (session + workflow wrapper)
│   ├── tools.py         # Calculator + document tools
│   ├── prompts.py       # System and chat prompts
│   ├── retrieval.py     # Simulated document retriever
│   ├── schemas.py       # Pydantic models (AnswerResponse, UserIntent, etc.)
│   └── main.py          # CLI runner for the assistant
└── tests/
    ├── __init__.py
    ├── test_schemas.py
    ├── test_tools_calculator.py
    ├── test_retrieval.py
    └── test_prompts.py

```
## Agent Architecture

The LangGraph agent follows this workflow:

![doc_assistant_agent](https://github.com/shilpamadini/document-assitant-agent/blob/c97130dab0ba9f49d0b249733c5b22f9ebf4e146/project/docs/langgraph_agent_architecture.png)

## Summary

### Intent Classification

Uses a prompt + structured output (UserIntent) to classify:
"qa"
"summarization"
"calculation"
"unknown"

### LangGraph Workflow
classify_intent
   ├── qa_agent
   ├── summarization_agent
   ├── calculation_agent
   ↓
update_memory
   ↓
END

### State Persistence

Using:
workflow.compile(checkpointer=InMemorySaver())
Each turn updates conversation_summary, active_documents, and actions_taken.

### Tools

All tools follow the @tool decorator pattern and log calls using ToolLogger.

### Pydantic Schemas

All agent outputs use structured schemas :AnswerResponse , SummarizationResponse, CalculationResponse, UpdateMemoryResponse to ensure predictable, structured, testable outputs.
