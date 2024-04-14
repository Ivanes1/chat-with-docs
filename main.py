from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    VectorStoreIndex,
    PromptTemplate,
)
from llama_index.llms.cohere import Cohere
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.embeddings.cohere import CohereEmbedding

COHERE_API_KEY = "<your-cohere-api-key>"

# LLM
llm = Cohere(api_key=COHERE_API_KEY)

# Postprocessor
cohere_rerank = CohereRerank(api_key=COHERE_API_KEY)

# Load the documents
input_dir_path = "./data"

loader = SimpleDirectoryReader(
    input_dir=input_dir_path,
    required_exts=[".pdf"],
    recursive=True,
)

docs = loader.load_data()

# Load the embedding model
embed_model = CohereEmbedding(
    cohere_api_key=COHERE_API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_query",
)

# Indexing and storing
Settings.embed_model = embed_model
index = VectorStoreIndex.from_documents(docs)

# Query engine
Settings.llm = llm
query_engine = index.as_query_engine(
    similarity_top_k=10, node_postprocessors=[cohere_rerank]
)

# Prompt template
qa_prompt_templ_str = (
    "Context information is below\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information above I want you \n"
    "to think step by step to answer the query in a crisp \n"
    "and clear manner. In case you don't know the answer say \n"
    "'I don't know!'.\n"
    "Query: {query_str}\n"
    "Answer: "
)


qa_prompt_templ = PromptTemplate(qa_prompt_templ_str)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_templ})

response = query_engine.query("What is 'query' and 'key' in the context of the paper?")
print(response)
print(response.get_formatted_sources(length=200))
