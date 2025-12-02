# Hierarchical Bookmark Retrieval for LLMs

A spec for improving context retrieval in long AI conversations.

## The Problem

Long conversations with LLMs degrade. The AI forgets what you told it 50 messages ago. You repeat yourself. It loses track of decisions you made together. Context windows fill up with noise instead of signal.

## The Idea

Instead of treating all conversation history equally, periodically generate **bookmarks**—LLM-judged summaries of what actually mattered:

- Key conclusions and decisions
- Corrections (what the *right* understanding turned out to be)
- Validated moments ("yes, exactly")
- Foundational context the user provided

When retrieving context for a new query, search bookmarks first, then fall back to standard similarity search.

## What's in the Spec

- **Data structures** for bookmarks and conversation chunks
- **Trigger logic** (when to generate bookmarks)
- **Bookmark generation** via single LLM pass
- **Hierarchical retrieval** algorithm
- **Validation strategy** with explicit decision gates
- **Cost analysis** and break-even conditions

## Key Design Choices

**Lazy, not eager.** No per-message processing. Bookmarks are generated at refresh time (manual trigger or threshold).

**Validation-first.** The spec includes a baseline test: if simple recency-weighting performs within 10% of the bookmark system, don't build it. Kill the project early if it's not worth the complexity.

**Honest about scope.** This helps power users with long working sessions. It doesn't matter for short conversations or casual Q&A.

## When to Build This

- Long conversations (50+ messages) are common in your product
- Users visibly repeat themselves or say "as I mentioned"
- You already have RAG infrastructure

## When NOT to Build This

- Most conversations are short
- Simple recency weighting is "good enough"
- Engineering time is better spent elsewhere

## Status

This is a **spec, not an implementation**. It's detailed enough to hand to an engineer, but it hasn't been built or validated yet.

## Background

I wrote this with no ML engineering experience, using AI (Claude) to critique and refine it through multiple versions. The result feels real—but I genuinely don't know if it is.

**I'm looking for feedback:**
- What's naive?
- What's already been done better?
- What's missing?
- Would you actually build this?

If you work on RAG, memory systems, or LLM infrastructure, I'd appreciate your eyes on it.

## Files

- `hierarchical_bookmark_retrieval_spec_2.md` — The full spec (v2)

## License

MIT — do whatever you want with it.

## Contact

Open an issue or reach out if you have thoughts.
