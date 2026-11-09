# LangChain Tools

An AI research assistant built with Streamlit, LangChain, and LangGraph. It can search the web with DuckDuckGo, find research papers on arXiv, and retrieve information from a FAISS knowledge base built from Wikipedia's artificial intelligence article.

The repository also includes [`tools_agents.ipynb`](tools_agents.ipynb), a notebook exploring LangChain tools and agents.

## Requirements

- Python
- An OpenAI API key
- Internet access for the OpenAI API, web search, arXiv, and loading the Wikipedia source

## Setup

From the project directory, create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the app

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, then enter your OpenAI API key in the sidebar. The app uses the `gpt-4.1-mini` model.

## Keep credentials private

Do not commit API keys or `.env` files. The app asks for the OpenAI API key in its sidebar; `.gitignore` excludes local `.env` files.
