import os
import textwrap
import tiktoken
import asyncio
import whisper
import openai
from typing import List, cast
from openai.types.chat import ChatCompletionMessageParam

# ========== CONFIG ==========
AUDIO_FILE = "The Knowledge Project with Shane Parrish-episode-#221 Bruce Flatt on Value, Discipline, and Durability.mp3"
CHUNK_SIZE = 4000  # characters per chunk
SUMMARY_STYLE = "structured"  # or "tl;dr"
OUTPUT_FILE = "podcast_summary.md"
OUTPUT_TRANSCRIPTION = "podcast_transcription.md"
OPENAI_MODEL = "gpt-4"
MAX_CONCURRENT_REQUESTS = 3
# ============================

# Load Whisper
model = whisper.load_model("base")
transcription_result = model.transcribe(
    "The Knowledge Project with Shane Parrish-episode-#221 Bruce Flatt on Value, Discipline, and Durability.mp3",
    verbose=True)
transcription_text = transcription_result["text"]

with open(OUTPUT_TRANSCRIPTION, "w", encoding="utf-8") as f:
    f.write(transcription_text)

open_ai_key = os.getenv("OPEN_API_KEY", "default_value")
# Set up OpenAI client (async)
client = openai.AsyncOpenAI(api_key=open_ai_key)


# Chunk text to stay within token limits
def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    return textwrap.wrap(text, max_chars, break_long_words=False, break_on_hyphens=True)

chunks = chunk_text(transcription_text, max_chars=CHUNK_SIZE)
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Async summarization of each chunk
async def summarize_chunk(chunk: str, idx: int) -> str:
    async with semaphore:
        print(f"📦 Summarizing chunk {idx + 1}/{len(chunks)}...")

        messages = cast(List[ChatCompletionMessageParam], [
            {
                "role": "system",
                "content": "You are a professional summarizer helping create structured knowledge notes from a podcast transcript. Your summaries are rich with insight and helpful for future reference."
            },
            {
                "role": "user",
                "content": f"""
Summarize the following transcription in detail for someone who wants to retain the knowledge discussed. This is not a TL;DR — create a set of structured notes that preserve nuance.

Instructions:
- Include all key insights and themes discussed.
- Quote the speaker if a line is especially impactful or clear.
- Organize ideas under subheadings.
- Make it useful to review later for someone who listened to the episode.
- Avoid fluff or vague summaries.

Transcription:
{chunk}
"""
            }
        ])

        try:
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()

        except openai.RateLimitError:
            print(f"⚠️ Rate limit hit. Retrying chunk {idx + 1} after 10s...")
            await asyncio.sleep(30)
            return await summarize_chunk(chunk, idx)

# Final merge of all summaries
async def combine_summaries(summaries: List[str]) -> str:
    print("📚 Merging all summaries into one detailed document...")

    messages = cast(List[ChatCompletionMessageParam], [
        {
            "role": "system",
            "content": "You are a summarizer merging detailed podcast note sections into a clean final set of podcast notes."
        },
        {
            "role": "user",
            "content": f"""Please merge and organize the following structured summaries into one cohesive Markdown document.

Summaries:
{chr(10).join(summaries)}
"""
        }
    ])

    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

def num_tokens_from_string(string: str, model_name: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

# Main async runner
async def main():
    chunk_tasks = [summarize_chunk(chunk, idx) for idx, chunk in enumerate(chunks)]
    detailed_chunks = await asyncio.gather(*chunk_tasks)

    combined_text = "\n\n".join(detailed_chunks)
    total_tokens = num_tokens_from_string(combined_text)

    print(f"📏 Combined token count: {total_tokens}")

    if total_tokens < 7500:
        print("📚 Merging with GPT-4...")
        final_summary = await combine_summaries(detailed_chunks)
    else:
        print("⚠️ Skipping GPT-4 merge (too large). Saving chunked summaries directly.")
        final_summary = combined_text  # just join without extra polish

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print(f"\n✅ Done! Summary saved to: {OUTPUT_FILE}")

# Run the pipeline
if __name__ == "__main__":
    asyncio.run(main())