from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import Settings

PROMPT_TEMPLATE = """
You are an expert assistant who answers questions based on the provided context.

- Use the context as your primary source of information.
- If the context contains the answer, respond directly using that information.
- If the context does not contain enough information, supplement with your general knowledge, but clearly indicate when you are doing so.
- If you don't know the answer or it is not in the document, say "I don't know" or "I have not found the information".
- ALWAYS respond in the same language as the question (English).

Context:
{context}

Question: {question}
Answer:
"""


def get_llm(settings: Settings) -> OllamaLLM:
    """
    Initializes and returns the FastFlowLM language model instance.

    Args:
        settings: Application settings.

    Returns:
        An instance of OllamaLLM.
    """
    return OllamaLLM(
        model=settings.llm_model,
        base_url=settings.fastflowlm_base_url,
        temperature=0.0,
    )


def build_qa_chain(
    llm: OllamaLLM,
    retriever: VectorStoreRetriever,
) -> RetrievalQA:
    """
    Builds and returns a RetrievalQA chain.

    Args:
        llm: The language model instance.
        retriever: The vector store retriever instance.

    Returns:
        A RetrievalQA chain configured with the custom prompt.
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

