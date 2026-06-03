from typing import List
from ingestion.document import Document


class PromptBuilder:
    @staticmethod
    def build_rag_prompt(query: str, contexts: List[Document]) -> str:
        """
        Combines the retrieved document chunks and the user query into a 
        strict, structured prompt template for the LLM.
        """
        context_block = ""
        for i, doc in enumerate(contexts, 1):
            source_name = doc.metadata.get("file_name", f"Unknown Source")
            context_block += f"--- Context Chunk {i} (Source: {source_name}) ---\n"
            context_block += f"{doc.text}\n\n"

        # Assemble the final instruction template
        prompt = f"""You are a helpful assistant. Answer the user's question accurately using ONLY the provided text contexts below. 
If the context does not contain the answer, say "I cannot find the answer in the provided documents." Do not make up information.

RELEVANT CONTEXTS:
{context_block}
USER QUESTION: {query}

YOUR ANSWER:"""
        
        return prompt