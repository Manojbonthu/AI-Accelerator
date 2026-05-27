"""
core/ingestion/gemma_client.py

Shared Gemini API client for describing images.
Uses Gemma 4 (free tier) with automatic retry on rate limits.
"""

import os
import time
import re
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Default model – Gemma 4 with 1,500 free requests/day
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")


def describe_image_with_gemma(image_bytes: bytes) -> Optional[str]:
    """Send image to Gemini and return a text description. Retries on 429 errors."""
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("    WARNING: GOOGLE_API_KEY not found in .env")
            return None

        client = genai.Client(api_key=api_key)

        prompt = """
        Describe this technical diagram or image in detail. Include:
        - All visible labels, annotations, and text
        - The relationship between different components
        - Any numerical values, units, or specifications shown
        - The overall purpose or system being illustrated

        If this is not a diagram but a photo or logo, describe what you see.
        If the image is blank or unreadable, say so.
        Keep the description under 500 words.
        """

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

        # Retry loop with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=GEMMA_MODEL,
                    contents=[prompt, image_part]
                )
                return response.text
            except Exception as e:
                err_str = str(e)
                if "429" in err_str and "retryDelay" in err_str:
                    match = re.search(r"retryDelay['\"]: '(\d+)s'", err_str)
                    delay = int(match.group(1)) if match else 5 * (attempt + 1)
                    print(f"    Rate limited. Waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    raise e

        return None

    except Exception as e:
        print(f"    Gemini API error: {e}")
        return None