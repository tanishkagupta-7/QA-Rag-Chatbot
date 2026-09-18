import json

from src.retriever import get_retriever


TEST_FILE = "evaluation/test_questions.json"


def load_test_questions():
    """Load evaluation questions."""
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_retrieval():
    """Evaluate whether expected pages appear in retrieved chunks."""

    test_questions = load_test_questions()
    retriever = get_retriever()

    total = len(test_questions)
    passed = 0

    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    for i, item in enumerate(test_questions, start=1):

        question = item["question"]
        expected_pages = item["expected_pages"]

        documents = retriever.invoke(question)

        retrieved_pages = []

        for document in documents:
            page = document.metadata.get("page_label", "Unknown")

            if page not in retrieved_pages:
                retrieved_pages.append(page)

        if not expected_pages:
            result = "N/A"
        else:
            found_pages = set(retrieved_pages).intersection(
                set(expected_pages)
            )

            if found_pages:
                passed += 1
                result = "PASS"
            else:
                result = "FAIL"

        print(f"\nTest {i}/{total}")
        print("-" * 70)
        print(f"Question: {question}")
        print(f"Expected pages: {expected_pages}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Result: {result}")

    evaluated_questions = sum(
        1 for item in test_questions
        if item["expected_pages"]
    )

    accuracy = (
        passed / evaluated_questions * 100
        if evaluated_questions
        else 0
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Evaluated questions: {evaluated_questions}")
    print(f"Passed: {passed}")
    print(f"Failed: {evaluated_questions - passed}")
    print(f"Retrieval accuracy: {accuracy:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_retrieval()