from app.services.local_answer_service import (
    build_document_profile,
    contextualize_locally,
    expand_queries_locally,
    generate_local_answer,
    stream_text,
)
from app.rag.chain import (
    _history_text,
    _overview_context_order,
    _merge_standalone_citation_lines,
    _uses_extractive_answer,
    clean_generated_answer,
    ensure_source_citations,
)
from app.core.config import settings


def test_advanced_tasks_use_ollama_even_when_quick_answers_are_extractive(monkeypatch):
    monkeypatch.setattr(settings, "GENERATION_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "QUICK_MODE_LOCAL", True)

    assert _uses_extractive_answer("quick", "answer") is True
    assert _uses_extractive_answer("quick", "quiz") is False
    assert _uses_extractive_answer("quick", "translate") is False


def test_standalone_model_citation_is_attached_to_previous_claim():
    answer = _merge_standalone_citation_lines(
        "- Databases: MongoDB.\n\n[Source 1]"
    )

    assert answer == "- Databases: MongoDB. [Source 1]\n"


def test_attached_short_claim_does_not_duplicate_source_label():
    results = [
        {
            "text": "The supported database is MongoDB.",
            "metadata": {"page": 1},
            "relevance": 0.9,
        }
    ]

    answer = ensure_source_citations(
        "- Databases: MongoDB.\n\n[Source 1]",
        1,
        results,
    )

    assert answer.count("[Source 1]") == 1
    assert "MongoDB. [Source 1]" in answer


def test_local_answer_is_grounded_and_cited():
    results = [
        {
            "text": (
                "Object-oriented programming uses objects and classes to organize software. "
                "Encapsulation combines data with the methods that operate on it."
            ),
            "relevance": 0.91,
            "metadata": {"page": 4},
        }
    ]

    answer = generate_local_answer(
        "What is object-oriented programming?",
        results,
        "balanced",
    )

    assert "objects and classes" in answer
    assert "[Source 1]" in answer
    assert "not found" not in answer.lower()


def test_local_follow_up_and_streaming_helpers():
    rewritten = contextualize_locally(
        "What are its advantages?",
        [{"role": "user", "content": "Explain polymorphism."}],
    )
    variants = expand_queries_locally("object oriented programming advantages")
    streamed = "".join(stream_text("A short locally generated answer with citations."))

    assert "polymorphism" in rewritten.lower()
    assert variants
    assert streamed == "A short locally generated answer with citations."


def test_missing_model_citations_are_added():
    answer = ensure_source_citations("A grounded answer without labels.", 3)

    assert "[Source 1]" in answer
    assert "[Source 3]" in answer


def test_invalid_model_citations_are_removed():
    answer = ensure_source_citations(
        "The supported point is cited [Source 2]. A bad label follows [Source 99].",
        3,
    )

    assert "[Source 2]" in answer
    assert "[Source 99]" not in answer


def test_generated_claims_receive_best_matching_source():
    answer = ensure_source_citations(
        "Mongoose provides schema validation for MongoDB applications.",
        2,
        [
            {"text": "Mongoose provides schemas and validation for MongoDB data."},
            {"text": "Express handles HTTP routes and middleware."},
        ],
    )

    assert answer.endswith("[Source 1]")


def test_each_generated_sentence_receives_its_best_source():
    answer = ensure_source_citations(
        "The degree is Computer Science. The CGPA is 9.12.",
        1,
        [{"text": "B.Tech in Computer Science with CGPA 9.12."}],
    )

    assert answer.count("[Source 1]") == 2


def test_small_model_meta_output_is_cleaned():
    answer = clean_generated_answer(
        "Mongoose is a **object data modeling** library.\n\n"
        "**Source citations**: [Source 1].\n"
        "**Word count**: 8 words."
    )

    assert answer == "Mongoose is an **object data modeling** library."


def test_small_model_confidence_meta_sentences_are_removed():
    answer = clean_generated_answer(
        "The candidate has a B.Tech with CGPA 9.12. "
        "There is no ambiguity in this information. "
        "The degree and CGPA are fully supported by the excerpt."
    )

    assert answer == "The candidate has a B.Tech with CGPA 9.12."


def test_overview_answer_filters_editor_noise():
    results = [
        {
            "text": (
                "EXPLORER OUTLINE TIMELINE node_modules package-lock.json Spaces:4 UTF-8 "
                "random toolbar labels and broken screen text."
            ),
            "relevance": 0.99,
            "metadata": {"page": 2, "filename": "backend-course.pdf"},
        },
        {
            "text": (
                "The course introduces backend development with Node.js and Express. "
                "It explains REST APIs, MongoDB persistence, authentication, and MVC architecture."
            ),
            "relevance": 0.85,
            "metadata": {"page": 5, "filename": "backend-course.pdf"},
        },
    ]

    answer = generate_local_answer("Summarize this PDF", results, "quick")

    assert "backend development" in answer.lower()
    assert "EXPLORER" not in answer


def test_local_answer_rejects_ocr_symbol_debris():
    results = [
        {
            "text": (
                "The MERN stack uses MongoDB and Express.js n NS] Fs] Ce "
                "————<——=* Mongoose."
            ),
            "relevance": 0.95,
            "metadata": {"page": 3},
        },
        {
            "text": (
                "Controllers implement application business logic and handle user requests."
            ),
            "relevance": 0.85,
            "metadata": {"page": 4},
        },
    ]

    answer = generate_local_answer("What do controllers do?", results, "quick")

    assert "Controllers implement" in answer
    assert "NS]" not in answer


def test_resume_overview_accepts_action_bullets_and_detects_structure():
    results = [
        {
            "text": (
                "EDUCATION\nB.Tech in Computer Science\n\n"
                "SKILLS\nPython, FastAPI, PostgreSQL\n\n"
                "PROJECTS\nBuilt a document search application with cited answers."
            ),
            "relevance": 0.82,
            "metadata": {"page": 1, "filename": "candidate.pdf"},
        },
        {
            "text": (
                "WORK EXPERIENCE\n"
                "Developed secure APIs and optimized database query performance."
            ),
            "relevance": 0.88,
            "metadata": {"page": 1, "filename": "candidate.pdf"},
        },
    ]

    profile = build_document_profile(results)
    answer = generate_local_answer("Summarize this PDF", results, "quick")

    assert profile["type"] == "résumé/CV"
    assert "résumé/CV" in answer
    assert "could not find" not in answer.lower()
    assert "[Source" in answer


def test_overview_context_starts_at_the_beginning_of_each_document():
    ranked = [
        {
            "text": "later",
            "relevance": 0.95,
            "metadata": {
                "document_id": "a",
                "page": 2,
                "page_chunk_index": 1,
            },
        },
        {
            "text": "beginning",
            "relevance": 0.80,
            "metadata": {
                "document_id": "a",
                "page": 1,
                "page_chunk_index": 0,
            },
        },
    ]

    ordered = _overview_context_order(ranked, 2)

    assert [result["text"] for result in ordered] == ["beginning", "later"]


def test_conversation_history_respects_context_budget(monkeypatch):
    monkeypatch.setattr(settings, "MAX_HISTORY_CHARACTERS", 120)
    history = [
        {"role": "user", "content": "old context " * 30},
        {"role": "assistant", "content": "recent answer " * 30},
    ]

    rendered = _history_text(history)

    assert len(rendered) <= 120
    assert "Assistant:" in rendered
