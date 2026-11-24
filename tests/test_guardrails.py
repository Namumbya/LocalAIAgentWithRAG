from __future__ import annotations

import guardrails as g


def test_classify_intent():
    assert g.classify_intent("Draft an owner reply about late delivery") == g.OWNER_REPLY
    assert g.classify_intent("Write a reply to the worst complaints") == g.OWNER_REPLY
    assert g.classify_intent("What are the worst complaints?") == g.COMPLAINTS
    assert g.classify_intent("Summarize the rating distribution") == g.STATS
    assert g.classify_intent("Should I order the pepperoni pizza?") == g.RECOMMEND
    assert g.classify_intent("Is the vegan pizza any good?") == g.RECOMMEND
    assert g.classify_intent("What do people say about the crust?") is None
    assert g.classify_intent("Hi") is None


def test_dish_topic_and_issue_extraction():
    assert g.dish_topic("Should I order the BBQ chicken pizza?") == "BBQ chicken pizza"
    assert g.dish_topic("Is the vegan pizza any good?") == "vegan pizza"
    assert g.dish_topic("Summarize the ratings") == ""
    assert g.issue_from_question("Draft an owner reply about late delivery") == "late delivery"
    assert g.issue_from_question("Draft an owner reply") == "your recent experience"


def test_evidence_fits_intent():
    low = {"min_rating": 1, "max_rating": 2}
    assert g.evidence_fits_intent(g.COMPLAINTS, "filter_reviews_by_rating", low)
    assert not g.evidence_fits_intent(
        g.COMPLAINTS, "filter_reviews_by_rating", {"min_rating": 1, "max_rating": 5}
    )
    assert not g.evidence_fits_intent(g.COMPLAINTS, "search_reviews", {"query": "bad"})
    assert g.evidence_fits_intent(g.STATS, "get_rating_stats", {"min_rating": 1})
    assert g.evidence_fits_intent(g.RECOMMEND, "search_reviews", {"query": "x"})
    assert g.evidence_fits_intent(g.OWNER_REPLY, "search_reviews", {"query": "late"})
    assert g.evidence_fits_intent(g.OWNER_REPLY, "filter_reviews_by_rating", low)


def test_parse_reviews_handles_both_tool_formats():
    search = "[1] Rating: 5/5 | Date: 2024-01-01\nGreat crust\nCrispy and chewy.\n\n[2] Rating: ?/5 | Date: ?\nNo rating\nHmm."
    reviews = g.parse_reviews(search)
    assert reviews[0] == g.Review("Great crust", 5, "Crispy and chewy.")
    assert reviews[1].rating is None

    filtered = "Found 1 reviews rated 1-2.\n\nTitle: Late\nDate: 2024-01-02\nRating: 1/5\nReview: Waited an hour."
    assert g.parse_reviews(filtered) == [g.Review("Late", 1, "Waited an hour.")]


def test_missing_topic_answer_and_synonyms():
    buffalo = "[1] Rating: 5/5 | Date: d\nBuffalo chicken pizza\nSpicy."
    assert g.missing_topic_answer("BBQ chicken pizza", buffalo) is not None
    assert g.missing_topic_answer("barbecue chicken pizza", "Great BBQ chicken pizza") is None
    assert g.missing_topic_answer("pepperoni pizza", "The pepperoni curled into cups") is None
    # Vegetarian is not vegan.
    assert g.missing_topic_answer("vegetarian pizza", "The vegan pizza was great") is not None


def test_recommendation_verdict_comes_from_ratings():
    low = [g.Review("Soggy", 2, "pepperoni soggy"), g.Review("Bad", 2, "pepperoni burnt")]
    assert g.recommendation_answer("pepperoni", low, "").startswith("**No.**")
    mixed = [g.Review("Ok", 3, "pepperoni fine"), g.Review("Good", 4, "pepperoni good")]
    assert g.recommendation_answer("pepperoni", mixed, "").startswith("**Mixed.**")


def test_letter_answer_falls_back_when_model_refuses():
    letter = g.letter_answer("late delivery", "I'm sorry, I can't write that.")
    assert letter.startswith("Dear Customer,\n\nThank you for sharing your experience regarding late delivery.")
    assert "We are sorry that late delivery fell short" in letter
    assert letter.endswith("Best regards,\nRestaurant Owner")


def test_stats_answer_passes_through_unparseable_text():
    assert g.stats_answer("no stats here") == "no stats here"


def test_letter_answer_uses_completed_stems():
    model = "We are sorry that your pizza took 90 minutes to arrive.\nTo fix this, we will add a driver at peak hours."
    letter = g.letter_answer("late delivery", model)
    assert (
        "regarding late delivery. We are sorry that your pizza took 90 minutes to arrive. "
        "To fix this, we will add a driver at peak hours.\n\n"
    ) in letter


def test_letter_answer_rejects_third_person_body():
    model = "The restaurant will address the late delivery issue."
    letter = g.letter_answer("late delivery", model)
    assert "The restaurant will" not in letter
    assert "We are sorry that late delivery fell short" in letter


def test_letter_answer_does_not_repeat_stems_on_one_line():
    model = "We are sorry that the pizza was cold. To fix this, we will check oven temperatures."
    letter = g.letter_answer("cold pizza", model)
    assert letter.count("To fix this") == 1
    assert "We are sorry that the pizza was cold. To fix this, we will check oven temperatures." in letter


def test_letter_answer_skips_echoed_blank_stems():
    model = (
        "We are sorry that ...\nTo fix this, we will ...\n"
        "We are sorry that you waited 90 minutes.\nTo fix this, we will add drivers."
    )
    letter = g.letter_answer("late delivery", model)
    assert "We are sorry that you waited 90 minutes. To fix this, we will add drivers." in letter


def test_letter_answer_accepts_paraphrased_fix_stem():
    model = (
        "We are sorry that the late delivery caused dissatisfaction. "
        "To address this issue, we will ensure that similar delays are avoided."
    )
    letter = g.letter_answer("late delivery", model)
    assert (
        "We are sorry that the late delivery caused dissatisfaction. "
        "To address this issue, we will ensure that similar delays are avoided."
    ) in letter
