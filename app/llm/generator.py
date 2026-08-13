from transformers import pipeline


class Generator:
    def __init__(self):
        self.pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer ONLY using the provided context. "
                    "If the answer is not present in the context, say 'I could not find the answer in the uploaded documents.' "
                    "Answer in one or two sentences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\nQuestion: {question}"
                ),
            },
        ]

        result = self.pipe(
            messages,
            max_new_tokens=64,
            do_sample=False,
        )

        generated = result[0]["generated_text"]

        # Newer chat pipelines usually return a list of messages.
        if isinstance(generated, list):
            return generated[-1]["content"].strip()

        # Fallback for string output.
        return str(generated).strip()