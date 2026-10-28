
import csv
import json
import os
import urllib.request
import urllib.error
from datetime import datetime

# Project files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "chat_history.csv")
MODEL = "llama3.2:latest"

# Healthcare sample questions
SAMPLE_QUESTIONS = [
    "What is a balanced diet?",
    "Why is drinking water important?",
    "What are signs of dehydration?",
    "Why is sleep important?",
    "How can I maintain good hygiene?",
    "What are the benefits of exercise?",
    "What is blood pressure?",
    "What is diabetes?",
    "Why are vitamins important?",
    "Why should we wash our hands?",
    "What are common causes of headaches?",
    "How can I reduce stress?",
    "Why are fruits and vegetables important?",
    "What is first aid?",
    "How can infections spread?",
    "Why are health checkups important?",
    "What is body temperature?",
    "What are the benefits of walking?",
    "Why is oral hygiene important?",
    "When should someone seek medical help?",
    "What is mental well-being?",
    "What is the difference between bacteria and viruses?"
]

SYSTEM_PROMPT = """
You are HealthCareBot, an educational health information assistant.
Give simple and clear general health information.
Do not diagnose diseases or prescribe medicines.
Encourage users to consult qualified healthcare professionals.
For emergencies, advise contacting local emergency services.
If unsure, say so clearly.
"""

def save_history(timestamp, question, answer):
    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["timestamp", "question", "answer"])

        writer.writerow([timestamp, question, answer])


def ask_ollama(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result["message"]["content"]


def main():
    print("\n=== HealthCareBot ===")
    print("Educational health information assistant")
    print("Type 'samples' to see test questions.")
    print("Type 'exit' to quit.\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Chat ended. Conversation saved.")
            break

        if question.lower() == "samples":
            for i, item in enumerate(SAMPLE_QUESTIONS, 1):
                print(f"{i}. {item}")
            continue

        if not question:
            continue

        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        messages.append({"role": "user", "content": question})

        try:
            answer = ask_ollama(messages)
            messages.append({"role": "assistant", "content": answer})

        except (urllib.error.URLError, TimeoutError, OSError) as error:
            messages.pop()
            print("\nOllama is not connected or its model is unavailable.")
            print("This question was not answered by the AI.")
            print("Technical detail:", error)
            print("Start Ollama and install the model when internet is available.\n")
            continue

        print("\nHealthCareBot:", answer, "\n")
        save_history(timestamp, question, answer)


if __name__ == "__main__":
    main()
