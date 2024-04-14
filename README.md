# Chat With Docs

Quick setup guide for indexing and querying documents using Cohere's LLM with the LlamaIndex package.

## Setup

- Install the necessary Python packages.
- Set your `COHERE_API_KEY`.

## Usage

1. Place `.pdf` files in the `./data` directory.
2. Run the script to load and query documents.

## Example

```python
print(query_engine.query("What is 'query' and 'key' in the context of the paper?"))
```
