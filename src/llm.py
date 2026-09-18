import os

from dotenv import load_dotenv
from groq import Groq


# Load variables from .env
load_dotenv()


MODEL_NAME = "openai/gpt-oss-20b"


def get_llm():
    """Create and return the Groq client."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Check your .env file."
        )

    client = Groq(api_key=api_key)

    return client


def generate_response(prompt):
    """Send a prompt to the LLM and return its response."""

    client = get_llm()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful question-answering assistant. "
                    "Answer clearly and concisely."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_completion_tokens=1024,
    )

    return response.choices[0].message.content

def rewrite_query(question, chat_history):
    """Rewrite a follow-up question into a standalone question."""

    if not chat_history:
        return question

    history_text = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in chat_history
    )

    prompt = f"""
You are a query-rewriting assistant.

Rewrite the user's latest question into a standalone question
that can be understood without the conversation history.

Use the conversation history only to resolve references such as:
"it", "they", "this", "that", "its", etc.

Do not answer the question.
Return ONLY the rewritten question.

Conversation history:
----------------
{history_text}
----------------

Latest question:
{question}

Standalone question:
"""

    return generate_response(prompt).strip()

if __name__ == "__main__":
    test_prompt = "What is Natural Language Processing?"

    answer = generate_response(test_prompt)

    print("=" * 60)
    print("LLM TEST")
    print("=" * 60)
    print(f"\nQuestion: {test_prompt}")
    print(f"\nAnswer:\n{answer}")

