"""
COMP 472 - Artificial Intelligence - Summer 2026
Mini Project 1: Intelligent Support Assistant with Sentiment Analysis and Retrieval

An AI-powered student support assistant that combines:
- Natural Language Processing (NLP)
- Sentiment Analysis (HuggingFace Transformers)
- Semantic Search (SentenceTransformers + Cosine Similarity)
- Conversational loop with escalation logic
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity


class SupportAssistant:
    """AI-powered student support assistant with semantic search and sentiment analysis."""

    def __init__(self, knowledge_base_path: str):
        """
        Initialize the assistant by loading models and the knowledge base.

        Args:
            knowledge_base_path: Path to the CSV file containing questions and answers.
        """
        print("Initializing Student Support AI...")

        self.knowledge_base_path = knowledge_base_path
        self.questions = []
        self.answers = []
        self.embeddings = None
        self.conversation_history = []

        # Load the sentence embedding model (all-MiniLM-L6-v2 is lightweight and accurate)
        print("Loading sentence embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load a 3-class sentiment model: POSITIVE, NEUTRAL, NEGATIVE
        print("Loading sentiment analysis model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

        self.load_knowledge_base()
        self.generate_embeddings()
        print("Initialization complete.\n")

    def load_knowledge_base(self):
        """
        Load questions and answers from the CSV file using pandas.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            KeyError: If the CSV is missing required 'question' or 'answer' columns.
        """
        try:
            df = pd.read_csv(self.knowledge_base_path)
            self.questions = df["question"].tolist()
            self.answers = df["answer"].tolist()
            print(f"Loaded {len(self.questions)} Q&A entries from '{self.knowledge_base_path}'.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Knowledge base file '{self.knowledge_base_path}' not found. "
                "Please ensure the file exists in the same directory."
            )
        except KeyError as e:
            raise KeyError(
                f"Missing expected column {e} in the CSV. "
                "The file must have 'question' and 'answer' columns."
            )

    def generate_embeddings(self):
        """
        Generate sentence embeddings for all questions stored in the knowledge base.

        Embeddings are dense vector representations of text that capture semantic meaning,
        allowing us to compare questions by meaning rather than exact word matching.
        """
        print("Generating embeddings for knowledge base questions...")
        self.embeddings = self.embedding_model.encode(
            self.questions, convert_to_numpy=True
        )
        print(f"Generated embeddings matrix of shape: {self.embeddings.shape}")

    def find_best_answer(self, user_query: str):
        """
        Find the most semantically relevant answer to the user's query.

        Steps:
        1. Convert the user query into an embedding vector.
        2. Compute cosine similarity against all stored question embeddings.
        3. Return the answer paired with the highest similarity score.

        Args:
            user_query: The question asked by the user.

        Returns:
            A tuple (answer: str, similarity_score: float).
        """
        query_embedding = self.embedding_model.encode(
            [user_query], convert_to_numpy=True
        )
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        return self.answers[best_idx], best_score

    def analyze_sentiment(self, text: str):
        """
        Analyze the sentiment of the provided text.
        Uses a HuggingFace transformers pipeline with the Cardiff NLP
        RoBERTa model, which supports three classes: POSITIVE, NEUTRAL, NEGATIVE. 
        Default one only has POSITIVE and NEGATIVE, so we use a more nuanced model for better escalation logic.
        """
        result = self.sentiment_pipeline(text)[0]
        label = result["label"].upper()
        confidence = round(result["score"], 2)
        return label, confidence

    def should_escalate(self, label: str, confidence: float) -> bool:
        """
        Determine whether a human advisor should be recommended.
        Escalation is triggered when the user expresses strong negative sentiment
        (confidence > 0.9), indicating they need human support.
        Returns:
            True if escalation is recommended, False otherwise.
        """
        return label == "NEGATIVE" and confidence > 0.9

    def chat(self):
        """
        Run the interactive conversation loop.
        Continuously accepts user input, performs sentiment analysis and semantic
        search, and prints results until the user types 'quit'.
        """
        print("Welcome to Student Support AI")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Thank you for using Student Support AI. Goodbye!")
                break

            # Store the user message in conversation history
            self.conversation_history.append({"role": "user", "content": user_input})

            # Sentiment analysis
            label, confidence = self.analyze_sentiment(user_input)
            print(f"Sentiment: {label} ({confidence:.2f})")

            # Escalation check
            if self.should_escalate(label, confidence):
                print("Recommended escalation: Contact human advisor.")

            # Semantic search for the best matching answer
            answer, similarity = self.find_best_answer(user_input)
            print(f"Answer: {answer}\n")

            # Store the assistant's response in conversation history
            self.conversation_history.append({"role": "assistant", "content": answer})


def main():
    """Entry point: create and run the support assistant."""
    try:
        assistant = SupportAssistant("knowledge_base.csv")
        assistant.chat()
    except FileNotFoundError as e:
        print(f"Error loading knowledge base: {e}")
    except KeyError as e:
        print(f"Error reading knowledge base format: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
