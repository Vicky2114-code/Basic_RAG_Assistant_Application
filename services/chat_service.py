from typing import Generator, Optional
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from config import MAX_RETRIEVAL_DOCS, TEMPLATES


class ChatService:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def generate_response(self, question: str, document_hash: str, model_name: str = "mistral",
                          temperature: float = 0.3) -> Generator[str, None, None]:
        """Generate streaming response to user query"""
        try:
            # Configure LLM
            llm = Ollama(
                model=model_name,
                temperature=temperature
            )

            # Configure retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": MAX_RETRIEVAL_DOCS,
                    "score_threshold": 0.4,
                    "filter": {"metadata.document_hash": document_hash}
                }
            )

            # Prompt template
            qa_template = """Answer the question based on the document content below.
            Provide a clear, structured response with:
            1. A brief Summary Paragraph overview
            2. 3-5 key points as bullet points
            3. Any important details or numbers when relevant
            4. Also add like conclusion after complete the result
            5. Try to give depth answer 

            Context: {context}
            Question: {question}
            """

            # Create and run QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=qa_template,
                        input_variables=["context", "question"]
                    )
                }
            )

            # Stream response
            response = qa_chain({"query": question})
            answer = response["result"]

            # Format response
            formatted_answer = answer.replace("- ", "• ").replace("* ", "• ")
            yield formatted_answer

        except Exception as e:
            yield f"Error: {str(e)}"