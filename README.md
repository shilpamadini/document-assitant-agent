# Document Assistant Project Instructions

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

## Implementation Tasks

### 1. Schema Implementation (schemas.py)

#### Task 1.1: AnswerResponse Schema
Create a Pydantic model for structured Q&A responses with the following fields:
- `question`: The original user question (string)
- `answer`: The generated answer (string)
- `sources`: List of source document IDs used (list of strings)
- `confidence`: Confidence score between 0 and 1 (float)
- `timestamp`: When the response was generated (datetime)

**Purpose**: This schema ensures consistent formatting of answers and tracks which documents were referenced.

#### Task 1.2: UserIntent Schema
Create a Pydantic model for intent classification with these fields:
- `intent_type`: The classified intent ("qa", "summarization", "calculation", or "unknown")
- `confidence`: Confidence in classification (float between 0 and 1)
- `reasoning`: Explanation for the classification (string)

**Purpose**: This schema helps the system understand what type of request the user is making and route it to the appropriate agent.

### 2. Agent State Implementation (agent.py, assistant.py)

#### 2.1: AgentState Properties
The `AgentState` class is structured as follows
 structure:
- `user_input`: Current user input
- `messages`: Conversation messages with LangGraph message annotation
- `intent`: Classified user intent
- `next_step`: Next node to execute in the graph
- `conversation_summary`: Summary of recent conversation
- `active_documents`: Document IDs currently being discussed
- `current_response`: The response being built
- `tools_used`: List of tools used in current turn
- `session_id` and `user_id`: Session management
- `actions_taken`: List of agent nodes executed (to be added in Task 2.6)

#### 2.2: Intent Classification Function
The `classify_intent` function is the first node in the graph. Its purpose is to query the LLM, by providing both the user's input
and message history (if any exists) and instructing the LLM to classify the intent so that graph can direct the request to the appropriate node.

#### Task 2.3: Calculation Agent  & Summarization Agent 



#### Task 2.4: Complete the Update Memory Function
This function maintains conversation context and tracks document references across turns.

#### Task 2.5: Workflow Creation

**Graph Structure**:
```
classify_intent --> [qa_agent|summarization_agent|calculation_agent] --> update_memory --> END
```

#### Task 2.6: State and Memory Persistence

To practice using state reducers and persistent memory, extend `AgentState` and your workflow as follows:

1. Add `operator.add` reducer to the `actions_taken` field of the `AgentState` schema. It will accumulate the names of each agent node that runs during a turn. For example:
2. (From Task 2.5) Import and use the InMemorySaver from the correct langgraph packagea and compile the workflow with a checkpointer using `InMemorySaver`. A checkpointer persists state across invocations, so your assistant will remember prior state even if you invoke the workflow multiple times. Modify `create_workflow` to call `workflow.compile(checkpointer=InMemorySaver())`. You will need to import `InMemorySaver`.
3. In the `process_message` method in `assistant.py`, you must properly set the values of the `configurable` value within the `config` object. Specifically, you must set:
   - The `thread_id` to the current_sessions.session_id
   - The `llm` to the configured LLM instance
   - The `tools`

These additions will enable you to track the flow of the agent and experiment with persistent state. Refer back to the state management and memory demo exercises for examples.

### 3. Prompt Implementation (prompts.py)

#### Task 3.1: Chat Prompt Template
Complete the `get_chat_prompt_template` function in `prompts.py`:
1. Finishing implement the function so that it supports ALL the `intent_type` parameters which could be "qa", "summarization", or "calculation"
2. Review prompts.py so you are aware of all the prompts in the file then make sure the `get_chat_prompt_template` the function sets the system_prompt to the correct value based on the `intent_type` parameter.
> Make sure to use existing prompts already defined in the file (QA_SYSTEM_PROMPT, SUMMARIZATION_SYSTEM_PROMPT, CALCULATION_SYSTEM_PROMPT)

**Purpose**: This provides context-aware prompts for different types of tasks.


#### Task 3.2: Implement the CalculationSystemPrompt
Implement the `CALCULATION_SYSTEM_PROMPT` constant in `prompts.py`:
1. Write a system prompt for the calculation agent that instructs the LLM to:
- Determine the document that must be retrieved and retrieve it using the document reader tool
- Determine the mathematical expression to calculate based on the user's input
- Use the calculator tool to perform the calculation
2. Make sure the LLM uses the calculator tool for ALL calculations no matter how simple

### 4. Tool Implementation (tools.py)

#### Task 4.1: Calculator Tool
Implement the `create_calculator_tool` function that:
1. Uses the `@tool` decorator to create a LangChain tool
2. Takes a mathematical expression as input
3. Validates the expression for safety (only allow basic math operations)
4. Evaluates the expression using Python's `eval()` function
5. Logs the tool usage with the ToolLogger
6. Returns a formatted result string
7. Handles errors gracefully

## Key Concepts for Success

### 1. LangChain Tool Pattern
Tools are functions decorated with `@tool` that can be called by LLMs. They must:
- Have clear docstrings describing their purpose and parameters
- Handle errors gracefully
- Return string results
- Log their usage for debugging

### 2. LangGraph State Management
The state flows through nodes and gets updated at each step. Key principles:
- Always return the updated state from node functions
- Use the state to pass information between nodes
- The state persists conversation context and intermediate results

### 3. Structured Output
Use `llm.with_structured_output(YourSchema)` to get reliable, typed responses from LLMs instead of parsing strings.

### 4. Conversation Memory
The system maintains conversation via the InMemorySaver checkpointer:
- Storing conversation messages with metadata
- Tracking active documents
- Summarizing conversations
- Providing context to subsequent requests

## Testing Your Implementation

1. **Unit Testing**: Test individual functions with sample inputs
2. **Integration Testing**: Test the complete workflow with various user inputs
3. **Edge Cases**: Test error handling and edge cases

## Common Pitfalls to Avoid

1. **Missing Error Handling**: Always wrap external calls in try-catch blocks
2. **Incorrect State Updates**: Ensure you're updating and returning the state correctly
3. **Prompt Engineering**: Make sure your prompts are clear and specific
4. **Tool Security**: Validate all inputs to prevent security issues

## Expected Behavior

After implementation, your assistant should be able to:
- Classify user intents correctly
- Search and retrieve relevant documents
- Answer questions with proper source citations
- Generate comprehensive summaries
- Perform calculations on document data
- Maintain conversation context across turns

Good luck with your implementation! Remember to test thoroughly and refer to the existing working code for guidance on patterns and best practices.
