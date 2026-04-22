from langchain_openai import OpenAIEmbeddings

def get_embedding() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")