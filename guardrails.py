"""Deterministic guardrails around a small local model.

Small models (e.g. qwen2.5:0.5b) pick wrong tool arguments, copy prompt
examples, and give one-word answers. For the question types this demo knows
(complaints, rating stats, "should I order X?", owner replies) the code decides
which evidence is valid and builds the fixed parts of the answer itself; the
model only writes short free-text pieces. Everything here is pure (no model or
tool calls) so it is easy to test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

COMPLAINTS = "complaints"
STATS = "stats"
RECOMMEND = "recommend"
OWNER_REPLY = "owner_reply"

_OWNER_REPLY_HINTS = ("draft", "owner reply", "write a reply", "letter", "email", "respond to")
_COMPLAINT_HINTS = ("worst", "complaint", "1-2", "1 star", "negative", "bad review", "low rating")
_STATS_HINTS = (
    "distribution",
    "average rating",
    "rating stats",
    "how many stars",
    "overall rating",
    "breakdown",
)
_RECOMMENDATION_HINTS = ("should i", "order", "worth", "any good", "recommend", "try the")

_TOPIC_STOPWORDS = {
    "the", "and", "any", "good", "pizza", "pizzas", "order", "should",
    "about", "with", "food", "dish", "reviews", "review", "try", "get",
}
# Spellings that mean the same dish word; a match on any one counts.
_TERM_SYNONYMS: tuple[tuple[str, ...], ...] = (
    ("bbq", "barbecue", "barbeque", "barbq"),
    ("vegetarian", "veggie"),
    ("margherita", "margarita"),
    ("fries", "chips"),
    ("gluten", "glutenfree", "celiac", "coeliac"),
)
# "Should I order the pepperoni pizza?" -> "pepperoni pizza"
ORDER_TOPIC_PATTERN = re.compile(
    r"\b(?:order|try|get)\s+(?:the\s+|a\s+|an\s+)?([^?.!]+)", re.IGNORECASE
)
# "Is the vegan pizza any good?" -> "vegan pizza"
_IS_GOOD_PATTERN = re.compile(
    r"\bis\s+(?:the\s+|their\s+)?(.+?)\s+(?:any\s+good|good|worth\b)", re.IGNORECASE
)
# "Draft an owner reply about late delivery" -> "late delivery"
_ISSUE_PATTERN = re.compile(r"\b(?:about|regarding|for|on)\s+(?:the\s+)?([^?.!]+)", re.IGNORECASE)
_REFUSAL_PATTERN = re.compile(
    r"\b(i can't|i cannot|i can not|i'm sorry|i am sorry|unable to|as an ai)\b", re.IGNORECASE
)
_LETTER_FRAME_PREFIXES = (
    "dear",
    "thank you for sharing",
    "we value your feedback",
    "best regards",
    "kind regards",
    "regards",
    "sincerely",
    "restaurant owner",
    "[restaurant owner]",
)


@dataclass(frozen=True)
class Review:
    title: str
    rating: int | None
    text: str


# --------------------------------------------------------------------------- #
# Question classification
# --------------------------------------------------------------------------- #


def _has_any(text: str, hints: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def is_owner_reply_request(question: str) -> bool:
    return _has_any(question, _OWNER_REPLY_HINTS)


def is_recommendation_request(question: str) -> bool:
    return _has_any(question, _RECOMMENDATION_HINTS)


def classify_intent(question: str) -> str | None:
    """Return the guarded question type, or None to leave it to the agent."""
    if is_owner_reply_request(question):
        return OWNER_REPLY
    if _has_any(question, _COMPLAINT_HINTS):
        return COMPLAINTS
    if _has_any(question, _STATS_HINTS):
        return STATS
    if is_recommendation_request(question) and dish_topic(question):
        return RECOMMEND
    return None


def dish_topic(question: str) -> str:
    """The dish a "should I order X" / "is X any good" question is about ('' if none)."""
    for pattern in (ORDER_TOPIC_PATTERN, _IS_GOOD_PATTERN):
        match = pattern.search(question)
        if match:
            return match.group(1).strip()
    return ""


def issue_from_question(question: str) -> str:
    """The issue an owner reply is about ("late delivery"), or a neutral default."""
    match = _ISSUE_PATTERN.search(question)
    if match:
        return match.group(1).strip()
    return "your recent experience"


def evidence_fits_intent(intent: str, tool_name: str, args: dict[str, object]) -> bool:
    """Whether a tool call the model made is valid evidence for this question type."""
    if intent == STATS:
        return tool_name == "get_rating_stats"
    if intent == RECOMMEND:
        return tool_name == "search_reviews"
    max_rating = args.get("max_rating", 5)
    low_only = (
        tool_name == "filter_reviews_by_rating"
        and isinstance(max_rating, int)
        and max_rating <= 2
    )
    if intent == COMPLAINTS:
        return low_only
    if intent == OWNER_REPLY:
        return tool_name == "search_reviews" or low_only
    return False


# --------------------------------------------------------------------------- #
# Evidence parsing and topic checks
# --------------------------------------------------------------------------- #


def _to_rating(value: str) -> int | None:
    return int(value) if value.isdigit() else None


def parse_reviews(tool_result: str) -> list[Review]:
    """Parse reviews from search_reviews or filter_reviews_by_rating output."""
    reviews: list[Review] = []
    for block in tool_result.split("\n\n"):
        block = block.strip()
        search = re.match(r"\[\d+\] Rating: (\S+)/5[^\n]*\n([^\n]*)\n?(.*)", block, re.DOTALL)
        if search:
            rating, title, text = search.groups()
            reviews.append(Review(title.strip(), _to_rating(rating), text.strip()))
            continue
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if ": " in line
        )
        if "Title" in fields and "Review" in fields:
            rating = fields.get("Rating", "").split("/", 1)[0]
            reviews.append(Review(fields["Title"], _to_rating(rating), fields["Review"]))
    return reviews


def _topic_terms(topic: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-z]+", topic.lower())
        if len(word) >= 3 and word not in _TOPIC_STOPWORDS
    ]


def term_in_text(term: str, text: str) -> bool:
    """True if `term` or a known spelling variant starts a word in `text`."""
    variants = next((group for group in _TERM_SYNONYMS if term in group), (term,))
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(variant)}", lowered) for variant in variants)


def mentions_topic(topic: str, text: str) -> bool:
    terms = _topic_terms(topic)
    return not terms or all(term_in_text(term, text) for term in terms)


def missing_topic_answer(topic: str, tool_result: str) -> str | None:
    """Fixed answer when search results never mention the dish asked about.

    Semantic search always returns the nearest reviews, so "BBQ chicken pizza"
    can come back as buffalo chicken reviews; don't let the model recommend it.
    """
    if not topic or mentions_topic(topic, tool_result):
        return None
    return (
        f"None of the reviews mention {topic}, so I can't recommend it either way. "
        "Try asking about a dish the reviews do cover, such as the pepperoni or "
        "vegan pizza."
    )


def usable_model_text(text: str) -> str:
    """The model's text, or '' if it is empty or a refusal."""
    cleaned = text.strip()
    if not cleaned or _REFUSAL_PATTERN.search(cleaned):
        return ""
    return cleaned


def _review_bullet(review: Review, limit: int = 140) -> str:
    rating = f"{review.rating}/5" if review.rating is not None else "?/5"
    text = review.text if len(review.text) <= limit else review.text[: limit - 3] + "..."
    return f'- "{review.title}" ({rating}): {text}'


# --------------------------------------------------------------------------- #
# Answer builders (model writes only the short free-text parts)
# --------------------------------------------------------------------------- #


def stats_answer(tool_result: str) -> str:
    """Fully deterministic rating-distribution answer."""
    counts = {int(r): int(c) for r, c in re.findall(r"^(\d) stars: (\d+)$", tool_result, re.M)}
    average = re.search(r"Average rating: ([\d.]+)", tool_result)
    total = sum(counts.values())
    if not counts or total == 0:
        return tool_result
    lines = [
        f"**Rating distribution across {total} reviews**"
        + (f" (average {average.group(1)}/5):" if average else ":")
    ]
    for rating in sorted(counts, reverse=True):
        share = round(100 * counts[rating] / total)
        lines.append(f"- {rating} stars: {counts[rating]} ({share}%)")
    ranked = sorted(counts, key=lambda r: counts[r], reverse=True)
    if len(ranked) >= 2:
        top, second = ranked[0], ranked[1]
        lines.append(
            f"\nMost reviews are {top} stars; the next largest group is "
            f"{second} stars ({counts[second]} reviews)."
        )
    return "\n".join(lines)


def complaints_prompt(tool_result: str) -> tuple[str, str]:
    return (
        "You summarize customer complaints from the reviews given. Be specific.",
        f"Reviews:\n{tool_result}\n\n"
        "List the 2-3 most common complaint themes as bullet points. "
        "Each bullet: the theme, then one short concrete example from a review. "
        "Output only the bullet points.",
    )


def complaints_answer(model_text: str, tool_result: str) -> str:
    header = re.match(r"Found (\d+) reviews rated (\d+)-(\d+)(?: \(([^)]*)\))?", tool_result)
    if header:
        count, low, high, breakdown = header.groups()
        opening = f"**{count} reviews are rated {low}-{high} stars**" + (
            f" ({breakdown})." if breakdown else "."
        )
    else:
        opening = "**Low-rated reviews**"
    parts = [opening]
    themes = usable_model_text(model_text)
    if themes:
        parts.append(f"Common themes:\n{themes}")
    examples = parse_reviews(tool_result)[:4]
    if examples:
        parts.append("Examples:\n" + "\n".join(_review_bullet(r) for r in examples))
    return "\n\n".join(parts)


def recommendation_prompt(topic: str, reviews: list[Review]) -> tuple[str, str]:
    evidence = "\n".join(_review_bullet(r, limit=300) for r in reviews)
    return (
        "You summarize what customers said about one dish. One sentence. No verdict words.",
        f"Dish: {topic}\nReviews about it:\n{evidence}\n\n"
        f"In one sentence, summarize what these reviews say about the {topic}.",
    )


def relevant_reviews(topic: str, tool_result: str) -> list[Review]:
    return [
        review
        for review in parse_reviews(tool_result)
        if mentions_topic(topic, f"{review.title} {review.text}")
    ]


def recommendation_answer(topic: str, reviews: list[Review], model_text: str) -> str:
    """Verdict comes from the ratings; the model only adds a one-line summary."""
    rated = [r.rating for r in reviews if r.rating is not None]
    if rated:
        average = sum(rated) / len(rated)
        verdict = "Yes" if average >= 4 else "No" if average <= 2.5 else "Mixed"
        noun = "review mentions" if len(reviews) == 1 else "reviews mention"
        opening = (
            f"**{verdict}.** {len(reviews)} retrieved {noun} the {topic}, "
            f"averaging {average:.1f}/5."
        )
    else:
        opening = f"Here is what reviews say about the {topic}."
    parts = [opening]
    summary = usable_model_text(model_text)
    if summary:
        parts.append(summary)
    if reviews:
        parts.append("What reviewers said:\n" + "\n".join(_review_bullet(r) for r in reviews[:3]))
    return "\n\n".join(parts)


_SORRY_LEAD = "We are sorry that"
_FIX_LEAD = "To fix this, we will"


def letter_prompt(issue: str, tool_result: str) -> tuple[str, str]:
    # Small models complete sentence stems far more reliably than they write
    # free-form letters (which drift into third person or copy examples).
    return (
        "You help a restaurant owner apologize to customers. Speak as the owner (we).",
        f"Issue: {issue}\nCustomer reviews:\n{tool_result}\n\n"
        "Complete these two sentences. Use a specific detail customers described "
        f"about {issue}. Output only the two completed sentences.\n"
        f"{_SORRY_LEAD} ...\n"
        f"{_FIX_LEAD} ...",
    )


def _completed_stem(text: str, lead: str) -> str:
    """First real completion of `lead` (models sometimes echo the blank stem first)."""
    for match in re.finditer(rf"{re.escape(lead)}\s*([^\n]*)", text, re.IGNORECASE):
        # Keep only the first sentence: models often put both stems on one line.
        first_sentence = re.split(r"(?<=[.!?])\s+", match.group(1).strip())[0]
        rest = re.sub(r"\[[^\]]*\]", "", first_sentence).strip().strip(".… ")
        if rest:
            return f"{lead} {rest}."
    return ""


def _paraphrased_fix(text: str, exclude: str) -> str:
    """A "we will ..." sentence when the model reworded the fix stem."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = re.sub(r"\[[^\]]*\]", "", sentence).strip()
        if sentence and sentence not in exclude and re.search(r"(?i)\bwe will\b", sentence):
            return sentence if sentence.endswith((".", "!")) else f"{sentence}."
    return ""


def _clean_letter_body(text: str) -> str:
    usable = usable_model_text(text)
    sorry = _completed_stem(usable, _SORRY_LEAD)
    fix = _completed_stem(usable, _FIX_LEAD) or _paraphrased_fix(usable, exclude=sorry)
    if sorry or fix:
        return " ".join(part for part in (sorry, fix) if part)
    # Model ignored the stems: keep free text only if it speaks as the owner.
    kept = [
        line.strip()
        for line in usable.splitlines()
        if line.strip() and not line.strip().lower().startswith(_LETTER_FRAME_PREFIXES)
    ]
    body = " ".join(kept)
    body = re.sub(r"\[[^\]]*\]", "", body)  # placeholders like [issue]
    body = re.sub(r"^\s*\d+[.)]\s*|\s+\d+[.)]\s+", " ", body)  # "1." / "2." numbering
    body = " ".join(body.split())
    return body if re.match(r"(?i)(we|i|our)\b", body) else ""


def letter_answer(issue: str, model_text: str) -> str:
    """Fixed letter frame; the model only writes the apology and fix sentences."""
    body = _clean_letter_body(model_text) or (
        f"We are sorry that {issue} fell short of what you should expect from us. "
        "We are reviewing what went wrong with our team so it does not happen again."
    )
    return (
        "Dear Customer,\n\n"
        f"Thank you for sharing your experience regarding {issue}. {body}\n\n"
        "We value your feedback and hope to serve you better next time.\n\n"
        "Best regards,\n"
        "Restaurant Owner"
    )


def general_prompt(question: str, tool_name: str, tool_result: str) -> tuple[str, str]:
    return (
        "You answer from provided review evidence only. Be concise and specific.",
        f"Question: {question}\n\nEvidence from `{tool_name}`:\n{tool_result}\n\n"
        "Answer the question, citing 1-3 reviews. Only use reviews that are "
        "relevant to the question. Do not invent reviews.",
    )
