# Vector-Search-System-with-Chat-Interface

Create an application that implements a vector database from scratch, loads data from blogs.json, and provides a simple search interface for users to find the most relevant blogs.

Add a chat interface to this application. This chat is essentially a RAG (Retriever augmented generation) system which:

  1. Retrieves all relevant information for a user’s query
  2. Invokes a LLM with the data from step 1 and asks it to give a response to the user’s question based on the provided data.

Which LLM to use? (eg. GPT-5, Claude etc etc.)

Option with $0: https://groq.com/pricing


## How to Run

1. Install dependencies in backend/requirements.txt

  pip install -r requirements.txt

2. Run backend/app.py

  python3 backend/app.py

3. Open frontend/index.html

