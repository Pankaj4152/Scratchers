from typing import List, Tuple, Dict, Any
import re
import numpy as np
from google import genai
from ingestion.document import Document


class RetrievalMetrics:
    """Automatic metrics for retrieval quality"""
    
    @staticmethod
    def precision_at_k(retrieved_labels: List[int], k: int) -> float:
        """Of top-K results, how many were relevant?"""
        if k == 0:
            return 0.0
        relevant_in_top_k = sum(retrieved_labels[:k])
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(retrieved_labels: List[int], total_relevant: int, k: int) -> float:
        """Of all relevant results, how many are in top-K?"""
        if total_relevant == 0:
            return 0.0
        relevant_in_top_k = sum(retrieved_labels[:k])
        return relevant_in_top_k / total_relevant
    
    @staticmethod
    def mrr(retrieved_labels: List[int]) -> float:
        """What rank was the first relevant result?"""
        for rank, is_relevant in enumerate(retrieved_labels, 1):
            if is_relevant:
                return 1.0 / rank
        return 0.0
    
    @staticmethod
    def ndcg(retrieved_labels: List[int], k: int) -> float:
        """Normalized Discounted Cumulative Gain"""
        dcg = 0.0
        for position, relevance in enumerate(retrieved_labels[:k], 1):
            dcg += relevance / np.log2(position + 1)
        
        num_relevant = sum(retrieved_labels[:k])
        ideal_dcg = 0.0
        for position in range(1, min(num_relevant + 1, k + 1)):
            ideal_dcg += 1.0 / np.log2(position + 1)
        
        if ideal_dcg == 0:
            return 0.0
        return dcg / ideal_dcg


class LLMBasedEvaluator:
    """Uses Gemini to evaluate RAG output quality"""
    
    def __init__(self, client: genai.Client):
        self.client = client
        self.judge_model = "gemini-2.5-flash"
    
    def evaluate_groundedness(self, context: str, answer: str) -> float:
        """
        Evaluates if the answer is grounded in the provided context.
        Returns 0.0 (hallucinated) to 1.0 (fully grounded).
        """
        prompt = f"""You are an expert AI Quality Assurance Judge. Your task is to evaluate if a RAG system's Answer is fully supported by the provided Context.

CONTEXT:
{context}

SYSTEM ANSWER:
{answer}

Analyze the System Answer statement by statement. Check if every fact is found in the Context.
Output your final score as a single decimal number between 0.0 and 1.0, where 1.0 means perfectly grounded with zero hallucinations, and 0.0 means completely fabricated.

Your output format MUST strictly be exactly like this:
SCORE: [your numeric score]"""

        try:
            response = self.client.models.generate_content(
                model=self.judge_model,
                contents=prompt
            )
            
            match = re.search(r"SCORE:\s*([\d.]+)", response.text)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception as e:
            print(f"Error in groundedness evaluation: {e}")
            return 0.0
    
    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """
        Evaluates if the answer actually answers the query.
        Returns 0.0 (irrelevant) to 1.0 (perfectly relevant).
        """
        prompt = f"""You are an expert evaluator. Does this answer properly address the user's question?

QUESTION:
{query}

ANSWER:
{answer}

Rate how well the answer addresses the question on a scale of 0.0 to 1.0, where:
- 1.0 = Completely answers the question with all key points
- 0.5 = Partially answers, missing some details
- 0.0 = Doesn't answer the question at all

Output format:
SCORE: [your numeric score]"""

        try:
            response = self.client.models.generate_content(
                model=self.judge_model,
                contents=prompt
            )
            
            match = re.search(r"SCORE:\s*([\d.]+)", response.text)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception as e:
            print(f"Error in answer relevance evaluation: {e}")
            return 0.0


class RAGEvaluator:
    """Complete RAG evaluation combining retrieval + generation metrics"""
    
    def __init__(self, client: genai.Client = None):
        self.retrieval_metrics = RetrievalMetrics()
        self.llm_evaluator = LLMBasedEvaluator(client) if client else None
    
    def evaluate_retrieval(
        self,
        retrieved_docs: List[Document],
        expected_keywords: List[str],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality.
        
        Args:
            retrieved_docs: List of retrieved documents
            expected_keywords: Keywords that should be in relevant documents
            k: Top-K to evaluate
            
        Returns:
            Dictionary with P@K, R@K, MRR, NDCG scores
        """
        # Create labels: 1 if all keywords in doc, 0 otherwise
        labels = []
        for doc in retrieved_docs:
            doc_text = doc.text.lower()
            is_relevant = all(kw.lower() in doc_text for kw in expected_keywords)
            labels.append(1 if is_relevant else 0)
        
        total_relevant = sum(labels)
        
        return {
            "precision_at_k": self.retrieval_metrics.precision_at_k(labels, k),
            "recall_at_k": self.retrieval_metrics.recall_at_k(labels, max(total_relevant, 1), k),
            "mrr": self.retrieval_metrics.mrr(labels),
            "ndcg_at_k": self.retrieval_metrics.ndcg(labels, k)
        }
    
    def evaluate_generation(
        self,
        query: str,
        answer: str,
        context: str
    ) -> Dict[str, float]:
        """
        Evaluate generation quality using LLM-as-Judge.
        
        Args:
            query: Original user query
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Dictionary with groundedness and relevance scores
        """
        if not self.llm_evaluator:
            return {"groundedness": 0.0, "answer_relevance": 0.0}
        
        return {
            "groundedness": self.llm_evaluator.evaluate_groundedness(context, answer),
            "answer_relevance": self.llm_evaluator.evaluate_answer_relevance(query, answer)
        }
    
    def evaluate_rag_pipeline(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document],
        expected_keywords: List[str] = None,
        retrieval_weight: float = 0.3,
        generation_weight: float = 0.7
    ) -> Dict[str, Any]:
        """
        Comprehensive RAG evaluation combining retrieval and generation metrics.
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_docs: List of retrieved documents
            expected_keywords: Keywords for retrieval evaluation
            retrieval_weight: Weight for retrieval score in overall score
            generation_weight: Weight for generation score in overall score
            
        Returns:
            Complete evaluation report
        """
        # Retrieval metrics
        retrieval_metrics = self.evaluate_retrieval(retrieved_docs, expected_keywords or [])
        retrieval_score = np.mean(list(retrieval_metrics.values())) if retrieval_metrics else 0.0
        
        # Generation metrics
        context = "\n".join([doc.text for doc in retrieved_docs])
        generation_metrics = self.evaluate_generation(query, answer, context)
        generation_score = np.mean(list(generation_metrics.values())) if generation_metrics else 0.0
        
        # Combined score
        overall_score = (retrieval_weight * retrieval_score) + (generation_weight * generation_score)
        
        return {
            "query": query,
            "retrieval_metrics": retrieval_metrics,
            "retrieval_score": retrieval_score,
            "generation_metrics": generation_metrics,
            "generation_score": generation_score,
            "overall_score": overall_score,
            "num_chunks_retrieved": len(retrieved_docs),
            "context_length": len(context),
            "answer_length": len(answer)
        }
    
    def print_evaluation_report(self, eval_result: Dict[str, Any]):
        """Pretty print evaluation results"""
        print("\n" + "=" * 80)
        print("RAG EVALUATION REPORT")
        print("=" * 80)
        
        print(f"\nQuery: {eval_result['query']}")
        print(f"\nRETRIEVAL METRICS:")
        print(f"  Precision@5: {eval_result['retrieval_metrics'].get('precision_at_k', 0):.3f}")
        print(f"  Recall@5:    {eval_result['retrieval_metrics'].get('recall_at_k', 0):.3f}")
        print(f"  MRR:         {eval_result['retrieval_metrics'].get('mrr', 0):.3f}")
        print(f"  NDCG@5:      {eval_result['retrieval_metrics'].get('ndcg_at_k', 0):.3f}")
        print(f"  → Retrieval Score: {eval_result['retrieval_score']:.3f}")
        
        print(f"\nGENERATION METRICS:")
        print(f"  Groundedness:    {eval_result['generation_metrics'].get('groundedness', 0):.3f}")
        print(f"  Answer Relevance: {eval_result['generation_metrics'].get('answer_relevance', 0):.3f}")
        print(f"  → Generation Score: {eval_result['generation_score']:.3f}")
        
        print(f"\nOVERALL METRICS:")
        print(f"  Overall Score: {eval_result['overall_score']:.3f}")
        print(f"  Chunks Retrieved: {eval_result['num_chunks_retrieved']}")
        print(f"  Context Length: {eval_result['context_length']} chars")
        print(f"  Answer Length: {eval_result['answer_length']} chars")
        print("=" * 80 + "\n")