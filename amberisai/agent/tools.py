"""
tools.py
========
Agent tools:
  1. classify_query()     — Route the question to right response type
  2. build_system_prompt() — Build rich contextual prompt for DeepSeek
  3. confidence_analysis() — Interpret confidence scores
  4. safe_escalation()    — Safety guardrails check
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── NURSE AMBER SYSTEM PROMPT ─────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are Nurse Amber, a warm, calm, deeply knowledgeable infant care AI assistant for AmberisAI. You help stressed parents understand their baby's cries and symptoms with DETAILED, THOROUGH guidance.

CRITICAL RULES:
- NEVER diagnose medically (no "your baby has X disease")
- ALWAYS add "Not medical advice — consult your pediatrician" at the end
- ALWAYS use the baby's name, age, and allergy info throughout your response
- ALWAYS base advice on the actual detection data provided
- Give LONG, DETAILED responses — parents need thorough guidance, not short answers
- Be warm, reassuring and empathetic — parents are often scared and exhausted
- Use numbered steps and clear sections so advice is easy to follow at 3am

RESPONSE STRUCTURE (always follow ALL sections in full detail):

1. 🔍 WHAT THIS MEANS:
   - Explain exactly what the detection result means in plain simple language
   - Explain WHY babies cry this way or show this symptom
   - Mention the confidence level and what it means for reliability
   - If there are secondary signals, explain those too

2. ✅ WHAT TO DO RIGHT NOW (Step by Step):
   - Give at least 4-6 very specific, numbered action steps
   - Each step should be detailed with exact technique (e.g. "Hold baby at 45 degrees, support head, pat back in circular motion for 2-3 minutes")
   - Include timing (how long to try each thing)
   - Include what to watch for after each step
   - Tailor steps to the baby's exact age in days
   - If allergy is listed, always suggest allergy-safe alternatives

3. 📋 THINGS TO CHECK:
   - List 4-5 additional things to observe or check
   - Diaper, temperature, last feed time, position, clothing etc.
   - Any patterns to watch for

4. 🚨 WHEN TO CALL THE DOCTOR:
   - Give very specific warning signs with exact thresholds
   - e.g. "If crying persists more than 30 minutes after feeding", "If temperature exceeds 38°C (100.4°F)"
   - List at least 4-5 specific escalation triggers
   - For babies under 60 days, be extra cautious with escalation advice

5. 💡 TIPS FOR NEXT TIME:
   - 2-3 preventive tips based on the pattern detected
   - Feeding schedule suggestions, sleep tips, etc.

6. 💚 REASSURANCE:
   - End with a warm, encouraging paragraph
   - Remind them they are doing great
   - Normalize the situation for the baby's age

SAFETY LANGUAGE:
✅ "Cry pattern suggests hunger-like signals"
✅ "Consistent with discomfort patterns"
✅ "Consider trying..."
❌ "Your baby has reflux"
❌ "This means your baby is sick"
❌ "Give medicine X"

ALLERGY AWARENESS:
- If milk allergy listed: always suggest hypoallergenic formula, avoid dairy in nursing diet
- If nut allergy: flag nursing diet considerations
- Always flag if query relates to a known allergy

AGE-APPROPRIATE ADVICE:
- 0-30 days: Feeding every 2-3 hours, jaundice watch, temperature regulation
- 31-90 days: Growth spurts, colic peak at 6 weeks, sleep cycles forming
- 91-180 days: Teething begins, more alert, solid food readiness signs
- 180+ days: Sleep regression, separation anxiety, starting solids"""


def build_system_prompt(context: dict) -> str:
    """
    Build a rich, contextual system prompt injecting all baby data.
    """
    prompt_parts = [BASE_SYSTEM_PROMPT, "\n\n── CURRENT CONTEXT ──────────────────"]

    # Baby profile
    profile = context.get("baby_profile")
    if profile:
        name = profile.get("nickname", "the baby")
        age = profile.get("age_days", "unknown")
        allergies = profile.get("allergies", [])
        allergy_str = ", ".join(allergies) if allergies else "none"
        prompt_parts.append(f"""
BABY PROFILE:
- Name: {name}
- Age: {age} days old
- Known allergies: {allergy_str}
- Always address the baby as "{name}" in your response""")
    else:
        prompt_parts.append("\nBABY PROFILE: Not available (answer generally)")

    # Latest session analysis
    session = context.get("latest_session")
    if session:
        audio = session.get("audio_json")
        image = session.get("image_json")

        if audio:
            detected = audio.get("detected_condition") or audio.get("primary_condition", "unknown")
            confidence = audio.get("confidence", 0)
            secondary = audio.get("secondary_condition", "")
            all_probs = audio.get("all_probabilities", {})
            low_conf = audio.get("low_confidence_warning", False)

            probs_str = ", ".join([f"{k}: {v*100:.0f}%" for k, v in
                                   sorted(all_probs.items(), key=lambda x: -x[1])]) if all_probs else "N/A"

            prompt_parts.append(f"""
LATEST CRY ANALYSIS (session: {session.get('session_id', 'N/A')}):
- Primary detection: {detected.upper()} ({confidence*100:.1f}% confidence)
- Secondary signal: {secondary or 'none'}
- All probabilities: {probs_str}
- Low confidence warning: {'YES — be cautious in advice' if low_conf else 'No'}
- Confidence level: {confidence_analysis(confidence)}""")

        if image:
            img_detected = image.get("detected_condition", "unknown")
            img_confidence = image.get("confidence", 0)
            prompt_parts.append(f"""
LATEST SKIN ANALYSIS:
- Detection: {img_detected.upper()} ({img_confidence*100:.1f}% confidence)""")
    else:
        prompt_parts.append("\nLATEST ANALYSIS: No analysis run yet this session")

    # Trends
    trends = context.get("trends")
    if trends and trends.get("total_sessions", 0) > 0:
        prompt_parts.append(f"""
SESSION HISTORY & TRENDS:
- Total sessions: {trends['total_sessions']}
- Pattern: {trends.get('summary', 'No pattern yet')}
- Most common condition: {trends.get('most_common', 'N/A')}
- Use this history to personalise advice (e.g., "This has worked 3x before")""")
    else:
        prompt_parts.append("\nSESSION HISTORY: First session — no history yet")

    prompt_parts.append("\n── END CONTEXT ──────────────────\n")
    prompt_parts.append("Now answer the parent's question using ALL the above context. Be warm, specific, and practical.")

    return "\n".join(prompt_parts)


def classify_query(query: str) -> str:
    """
    Classify the parent's query into response types.
    Returns one of: explain | action | trend | escalate | summary | qa
    """
    q = query.lower().strip()

    # Escalation indicators
    escalation_words = ["emergency", "fever", "hospital", "doctor", "serious", "scared",
                        "not breathing", "blue", "seizure", "unconscious", "blood"]
    if any(w in q for w in escalation_words):
        return "escalate"

    # Trend questions
    trend_words = ["pattern", "trend", "week", "always", "often", "usually",
                   "history", "before", "last time", "how many"]
    if any(w in q for w in trend_words):
        return "trend"

    # Summary
    summary_words = ["summary", "today", "overall", "how has", "report", "overview"]
    if any(w in q for w in summary_words):
        return "summary"

    # Action questions
    action_words = ["what do i do", "what should i", "how do i", "help",
                    "what to do", "fix", "stop", "calm", "soothe", "feed"]
    if any(w in q for w in action_words):
        return "action"

    # Explanation questions
    explain_words = ["what is", "what does", "what does it mean", "why",
                     "explain", "mean", "confidence", "percentage", "normal"]
    if any(w in q for w in explain_words):
        return "explain"

    return "qa"  # default: general Q&A


def confidence_analysis(confidence: float) -> str:
    """Human-readable confidence interpretation."""
    if confidence >= 0.85:
        return "VERY HIGH — highly reliable detection, act on this"
    elif confidence >= 0.70:
        return "HIGH — reliable, proceed with confidence"
    elif confidence >= 0.55:
        return "MODERATE — likely correct, but watch for other signs"
    elif confidence >= 0.40:
        return "LOW — uncertain, observe carefully before acting"
    else:
        return "VERY LOW — unclear signal, look for visual cues instead"


def safe_escalation(query: str, context: dict) -> Optional[str]:
    """
    Check if query requires immediate medical escalation.
    Returns escalation message string if needed, None otherwise.
    """
    q = query.lower()

    critical_keywords = [
        "not breathing", "stopped breathing", "blue lips", "blue face",
        "unconscious", "won't wake", "seizure", "convulsion",
        "high fever", "very high fever", "very hot"
    ]

    if any(kw in q for kw in critical_keywords):
        return (
            "⚠️ IMPORTANT: Based on what you've described, please contact emergency services "
            "or take your baby to the nearest hospital immediately. Do not wait. "
            "Call your local emergency number now. This is not medical advice — "
            "but these symptoms require immediate professional evaluation."
        )

    # Check age-specific escalation
    profile = context.get("baby_profile")
    if profile:
        age = profile.get("age_days", 0)
        if age < 60:
            # Newborns need faster escalation
            newborn_keywords = ["fever", "temperature", "not eating", "won't eat",
                                "yellow", "jaundice", "dark urine"]
            if any(kw in q for kw in newborn_keywords):
                return None  # Not immediate emergency, but flag in response

    return None


def format_sse_event(data: str) -> str:
    """Format a Server-Sent Events message."""
    return f"data: {data}\n\n"


def build_messages_for_deepseek(context: dict, query: str) -> list:
    """
    Build the full messages array for DeepSeek API.
    Includes system prompt + chat history + current query.
    """
    system_prompt = build_system_prompt(context)

    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history (already formatted as {role, content})
    chat_history = context.get("chat_history", [])
    for msg in chat_history[-10:]:  # Last 10 messages for context window
        messages.append(msg)

    # Add current query
    messages.append({"role": "user", "content": query})

    return messages