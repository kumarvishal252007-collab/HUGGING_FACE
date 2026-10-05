import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=token)

def generate_flashcards(notes_text: str, count: int = 4):
    system_instruction = (
        "You are an expert study assistant. Your task is to turn study notes into effective flashcards. "
        "Strictly adhere to the provided format with no additional conversational filler."
    )
    
    prompt = f"""Create {count} distinct flashcards from the notes below.

Format each card strictly as:
---
Front: <Concise term, key process, or question>
Back: <Clear, direct definition, equation, or explanation>

Notes:
{notes_text}
"""

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    photosynthesis_notes = (
        "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy. "
        "Overall Chemical Equation: 6CO2 + 6H2O + Light Energy -> C6H12O6 + 6O2.\n"
        "Site of Reaction: Takes place in chloroplasts. Chlorophyll inside the thylakoid membranes absorbs blue and red wavelengths while reflecting green.\n"
        "Stage 1 - Light-Dependent Reactions: Occur in the thylakoid membrane. Uses light to split water (photolysis), releasing oxygen (O2) and generating ATP and NADPH.\n"
        "Stage 2 - Calvin Cycle (Light-Independent Reactions): Occurs in the stroma. Uses ATP and NADPH from stage 1 along with carbon dioxide (CO2) to synthesize glucose."
    )
    
    print("Generating Photosynthesis Flashcards...\n")
    try:
        flashcards = generate_flashcards(photosynthesis_notes, count=7)
        print(flashcards)
    except Exception as e:
        print(f"Error: {e}")