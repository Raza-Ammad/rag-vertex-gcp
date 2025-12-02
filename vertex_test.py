import vertexai
from vertexai.generative_models import GenerativeModel

# Your GCP project + region
PROJECT_ID = "starry-journal-480011-m8"
LOCATION = "us-central1"  # Region that supports the latest Gemini models


def main():
    # Initialise Vertex AI
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # ✅ Use a current, supported Flash model on Vertex
    model = GenerativeModel("gemini-2.5-flash")

    prompt = "In one sentence, explain what a Retrieval-Augmented Generation (RAG) system is."

    # Call the model
    response = model.generate_content(prompt)

    print("Vertex Gemini response:\n")
    print(response.text)


if __name__ == "__main__":
    main()