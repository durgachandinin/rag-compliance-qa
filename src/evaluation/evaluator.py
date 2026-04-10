from typing import List, Dict

class RAGEvaluator:
    def build_eval_dataset(self, qa_pairs: List[Dict], rag_chain):
        from datasets import Dataset
        questions, answers, contexts, ground_truths = [], [], [], []
        print(f"Evaluating {len(qa_pairs)} Q&A pairs...")
        for pair in qa_pairs:
            question = pair["question"]
            result = rag_chain.ask(question)
            questions.append(question)
            answers.append(result["answer"])
            contexts.append(result["source_chunks"])
            ground_truths.append(pair.get("ground_truth", ""))
            rag_chain.reset_memory()
        return Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })

    def evaluate_pipeline(self, qa_pairs: List[Dict], rag_chain) -> Dict:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision
        dataset = self.build_eval_dataset(qa_pairs, rag_chain)
        return evaluate(dataset, metrics=[answer_relevancy, faithfulness, context_recall, context_precision])

    def print_report(self, results: Dict) -> None:
        print("\n" + "=" * 55)
        print("RAG EVALUATION REPORT  (RAGAS)")
        print("=" * 55)
        metric_labels = {
            "answer_relevancy": "Answer Relevancy",
            "faithfulness": "Faithfulness (Anti-hallucination)",
            "context_recall": "Context Recall",
            "context_precision": "Context Precision",
        }
        thresholds = {"answer_relevancy": 0.85, "faithfulness": 0.90,
                      "context_recall": 0.80, "context_precision": 0.75}
        for key, label in metric_labels.items():
            if key in results:
                score = results[key]
                status = "PASS" if score >= thresholds[key] else "NEEDS IMPROVEMENT"
                print(f"  {label:<38s}: {score:.3f}  [{status}]")
        print("=" * 55)


COMPLIANCE_EVAL_QA_PAIRS: List[Dict] = [
    {"question": "What is the minimum CET1 ratio under Basel III?",
     "ground_truth": "The minimum CET1 ratio under Basel III is 4.5% of risk-weighted assets."},
    {"question": "What was Goldman Sachs LCR in 2023?",
     "ground_truth": "Goldman Sachs LCR was approximately 128% as of December 2023."},
    {"question": "What is the capital conservation buffer under Basel III?",
     "ground_truth": "The capital conservation buffer is 2.5% of CET1 capital above the regulatory minimum."},
    {"question": "What is JPMorgan G-SIB surcharge?",
     "ground_truth": "JPMorgan Chase G-SIB surcharge is 3.5%, applicable to CET1 capital."},
    {"question": "What does LCR stand for and what is its minimum requirement?",
     "ground_truth": "LCR stands for Liquidity Coverage Ratio. The minimum requirement is 100% under Basel III."},
    {"question": "How does Goldman Sachs CET1 ratio compare to the regulatory minimum?",
     "ground_truth": "Goldman Sachs CET1 was 14.5%, well above their total requirement of 10.9%."},
]
