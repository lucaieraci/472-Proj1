# COMP 472 Mini Project 1

Intelligent Support Assistant with Sentiment Analysis and Retrieval for COMP 472,
Summer 2026.

## What the Assistant Does

- Loads student-support questions and answers from `knowledge_base.csv`.
- Builds sentence embeddings for the knowledge-base questions.
- Uses cosine similarity to retrieve the closest answer for each user query.
- Detects `POSITIVE`, `NEUTRAL`, or `NEGATIVE` sentiment with a Hugging Face model.
- Recommends human-advisor escalation for strongly negative messages.
- Runs an interactive loop until the user types `quit`.

## Project Files

- `support_assistant.py` - main Python source code.
- `knowledge_base.csv` - question and answer knowledge base.
- `reflection.txt` - team reflection, contributions, difficulties, and time breakdown.
- `screenshots/` - screenshots showing startup, semantic retrieval, sentiment, escalation, and exit.
- `project1_Summer_2026.pdf` - assignment instructions.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run

```bash
python support_assistant.py
```

Example interaction:

```text
Welcome to Student Support AI
Type 'quit' to exit.

You: I cannot access my account and this is terrible
Sentiment: NEGATIVE (0.99)
Recommended escalation: Contact human advisor.
Answer: Visit the IT portal at it.concordia.ca and select Forgot Password.
```
