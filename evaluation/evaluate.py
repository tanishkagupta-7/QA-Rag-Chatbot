import json
import math

from src.rag_pipeline import answer_question
from src.embeddings import get_embeddings


TEST_FILE = "evaluation/test_questions.json"
SIMILARITY_THRESHOLD = 0.70


def load_test_questions():
    """Load evaluation questions from JSON."""
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def evaluate():
    """Evaluate RAG answers using semantic similarity."""

    test_questions = load_test_questions()

    print("=" * 70)
    print("RAG CHATBOT SEMANTIC EVALUATION")
    print("=" * 70)

    print("\nLoading embedding model...")
    embeddings = get_embeddings()

    total = len(test_questions)
    passed = 0

    for i, item in enumerate(test_questions, start=1):

        question = item["question"]
        expected_answer = item["expected_answer"]

        print(f"\nTest {i}/{total}")
        print("-" * 70)
        print(f"Question: {question}")

        answer, documents = answer_question(question)

        print(f"\nGenerated Answer:")
        print(answer)

        print(f"\nExpected Answer:")
        print(expected_answer)

        # Create embeddings for generated and expected answers
        generated_vector = embeddings.embed_query(answer)
        expected_vector = embeddings.embed_query(expected_answer)

        similarity = cosine_similarity(
            generated_vector,
            expected_vector,
        )

        # Collect source pages
        sources = []

        for document in documents:
            page = document.metadata.get(
                "page_label",
                "Unknown"
            )

            if page not in sources:
                sources.append(page)

        print(f"\nSources: {sources}")
        print(f"Semantic similarity: {similarity:.2%}")

        if similarity >= SIMILARITY_THRESHOLD:
            passed += 1
            print("Result: PASS")
        else:
            print("Result: FAIL")

    accuracy = passed / total * 100

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Semantic accuracy: {accuracy:.2f}%")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()