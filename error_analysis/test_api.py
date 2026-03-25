"""Diagnostic: see actual response with the real prompt."""
import os, json
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

api_key = os.environ.get("Gemini_API_Key", "")
from google import genai
client = genai.Client(api_key=api_key)

PROMPT = """You are an expert analyst. Your job is to evaluate evidence step by step, consider alternatives, and reach a justified conclusion."""

user_msg = """Context: I have been having problems with my internet connection dropping out every few minutes.
Reference: High
Incorrect Prediction: Medium"""

print("Test 1: WITH response_mime_type=application/json")
try:
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_msg,
        config=genai.types.GenerateContentConfig(
            system_instruction=PROMPT,
            response_mime_type="application/json",
            max_output_tokens=512,
        ),
    )
    print(f"  response.text[:200] = {repr(r.text[:200])}")
    print(f"  Parts: {len(r.candidates[0].content.parts)}")
    for i, p in enumerate(r.candidates[0].content.parts):
        t = getattr(p, 'thought', None)
        txt = p.text[:80] if p.text else 'None'
        print(f"    Part {i}: thought={t} text={repr(txt)}")
    # Try parsing
    try:
        parsed = json.loads(r.text)
        print(f"  JSON parse: OK")
    except Exception as je:
        print(f"  JSON parse error: {je}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")

print("\nTest 2: WITHOUT response_mime_type (plain text)")
try:
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_msg + "\n\nRespond ONLY with a valid JSON object, no other text.",
        config=genai.types.GenerateContentConfig(
            system_instruction=PROMPT,
            max_output_tokens=512,
        ),
    )
    raw = r.text.strip()
    print(f"  response.text[:200] = {repr(raw[:200])}")
    # Try to extract JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
        print(f"  JSON parse: OK")
    except Exception as je:
        print(f"  JSON parse error: {je}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")
