"""ReviewIQ — Ad Copy Generation Service"""

import re
import json
import anthropic
from collections import Counter
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the","a","an","is","it","this","i","my","and","or","but","was","are",
    "be","have","has","to","of","in","for","on","with","at","by","from","as",
    "not","that","they","we","you","he","she","so","if","no","do","just","very",
    "got","get","really","so","too","also","would","will","did","does","its",
    "product","item","use","used","buy","bought","ordered","order","amazon",
    "shipping","ship","arrived","delivery","star","stars","review","reviews",
}


def extract_power_phrases(texts: List[str], top_n: int = 20) -> List[str]:
    """
    Extract the most impactful phrases from positive reviews.
    Uses noun phrase patterns + frequency ranking.
    """
    # Simple bigram/trigram extraction
    all_ngrams = []
    for text in texts:
        text_clean = re.sub(r"[^a-z\s]", " ", text.lower())
        words = [w for w in text_clean.split() if w not in STOPWORDS and len(w) > 2]

        # Bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if not any(w in STOPWORDS for w in [words[i], words[i+1]]):
                all_ngrams.append(bigram)

        # Trigrams
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            all_ngrams.append(trigram)

    # Filter meaningful phrases
    phrase_counts = Counter(all_ngrams)
    meaningful = [
        phrase for phrase, count in phrase_counts.most_common(200)
        if count >= 3 and len(phrase.split()) >= 2
        and not all(w in STOPWORDS for w in phrase.split())
    ]

    return meaningful[:top_n]


async def generate_ad_copy_claude(
    asin: str,
    power_phrases: List[str],
    formats: List[str],
    brand_tone: Optional[str],
    api_key: str,
    model: str,
) -> List[dict]:
    """Call Claude to generate platform-specific ad copy from customer phrases."""

    format_instructions = {
        "amazon_bullets": "5 Amazon A+ content bullet points (start each with a benefit, max 100 chars each)",
        "facebook_ad": "3 Facebook/Instagram ad copy variants (hook + body + CTA, max 150 words each)",
        "instagram_caption": "1 Instagram caption with emoji and 5 relevant hashtags",
        "google_rsa": "5 headlines (max 30 chars) and 3 descriptions (max 90 chars) for Google RSA",
        "email_subject": "5 email subject line variants (max 50 chars each, test-worthy)",
    }

    selected_formats = {k: v for k, v in format_instructions.items() if k in formats}

    prompt = f"""You are a conversion copywriter specializing in Amazon and DTC brands.

## Product ASIN
{asin}

## Authentic Customer Phrases (extracted from {len(power_phrases)} top-rated reviews)
{chr(10).join(f'- "{p}"' for p in power_phrases[:15])}

## Brand Tone
{brand_tone or "Professional yet approachable"}

## Task
Generate ad copy for these formats using the customer's own language above.
The copy should feel authentic because it IS based on real customer words.

Return ONLY valid JSON:
{{
  {', '.join(f'"{fmt}": {{"copy": "...", "source_phrases": ["phrase1", "phrase2"]}}' for fmt in selected_formats)}
}}

Rules:
- Use customer phrases naturally, don't just list them
- Lead with the strongest benefit
- Keep copy scannable
- Match the brand tone throughout
"""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
        return [
            {
                "format": fmt,
                "copy": data.get("copy", ""),
                "source_phrases": data.get("source_phrases", []),
            }
            for fmt, data in parsed.items()
        ]
    except json.JSONDecodeError:
        logger.warning(f"Ad copy JSON parse failed: {raw[:200]}")
        return [{"format": "raw", "copy": raw, "source_phrases": []}]
