# Hierarchical Bookmark Retrieval System v2

## Overview

A lightweight retrieval layer that uses LLM-judged salience to improve context retrieval in long conversations. Triggered lazily—no per-message overhead.

**Core hypothesis:** An LLM reviewing a conversation can identify what mattered better than embedding similarity alone.

**Build time:** ~1 week

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                         │
│          [Refresh] button appears after threshold           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                       │
│   - Monitors conversation length                            │
│   - Triggers bookmark generation at threshold               │
│   - Routes queries through hierarchical retrieval           │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│       LLM        │ │ BOOKMARK STORE  │ │   RAW STORAGE   │
│   (existing)     │ │  (Vector DB)    │ │   (Vector DB)   │
└──────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## Data Structures

```python
@dataclass
class Bookmark:
    id: str
    content: str                    # 1-2 sentence summary
    source_chunk_ids: List[str]     # Pointers to raw content
    importance: float               # 0.0 - 1.0
    bookmark_type: str              # "conclusion" | "resolution" | "validated" | "key_claim"
    created_at: datetime
    last_relevant_at: datetime      # For decay calculation

@dataclass
class ConversationChunk:
    id: str
    content: str
    embedding: np.array
    timestamp: datetime
    speaker: str                    # "user" | "assistant"
```

---

## Algorithm 1: Trigger Condition

One line of logic. No classifiers, no threshold tuning.

```python
def should_show_refresh_button(conversation: List[ConversationChunk]) -> bool:
    # Option A: Message count
    if len(conversation) > 50:
        return True
    
    # Option B: Context window pressure
    total_tokens = sum(count_tokens(c.content) for c in conversation)
    if total_tokens > (CONTEXT_WINDOW_SIZE * 0.8):
        return True
    
    return False
```

---

## Algorithm 2: Bookmark Generation

Single LLM pass over the full conversation. No per-message processing.

```python
def generate_bookmarks(conversation: List[ConversationChunk]) -> List[Bookmark]:
    """
    One LLM call. Replaces all signal detection classifiers.
    """
    
    prompt = """
    Review this conversation and identify moments worth remembering for future reference.
    
    Extract the following (if present):
    
    1. KEY CONCLUSIONS
       Decisions made, answers reached, problems solved.
       
    2. RESOLUTIONS
       When the user corrected a misunderstanding, summarize what the CORRECT 
       understanding turned out to be. Do NOT summarize the error—summarize 
       what was established after the correction.
       
    3. VALIDATED UNDERSTANDING
       Moments where the user confirmed the assistant understood correctly.
       ("Yes exactly", "That's it", "Right")
       
    4. FOUNDATIONAL CONTEXT
       Information the user provided about themselves, their project, or their 
       constraints that would be relevant to future questions.
    
    For each, provide:
    - A 1-2 sentence summary (in your own words, not quoted)
    - Which message numbers it spans
    - Importance score (0.0 - 1.0)
    
    Output as JSON array.
    """
    
    response = llm_call(
        prompt=prompt,
        conversation=format_conversation_with_indices(conversation)
    )
    
    bookmarks = []
    for item in parse_json(response):
        bookmarks.append(Bookmark(
            id=generate_id(),
            content=item["summary"],
            source_chunk_ids=get_chunk_ids_for_range(item["message_range"], conversation),
            importance=item["importance"],
            bookmark_type=item["type"],
            created_at=now(),
            last_relevant_at=now()
        ))
    
    return bookmarks
```

**Key fix from peer review:** The prompt asks for *resolutions*, not corrections. We want "the user clarified they meant X" not "the user said we were wrong about Y."

---

## Algorithm 3: Hierarchical Retrieval

Query bookmarks first. Fall back to raw chunks.

```python
def retrieve_context(
    query: str,
    bookmarks: List[Bookmark],
    raw_chunks: List[ConversationChunk],
    context_budget: int = 4000  # tokens
) -> List[ConversationChunk]:
    
    query_embedding = embed(query)
    
    # === TIER 1: BOOKMARK SEARCH ===
    bookmark_scores = []
    for bookmark in bookmarks:
        similarity = cosine_similarity(query_embedding, embed(bookmark.content))
        
        # Weight by importance
        score = similarity * (0.5 + 0.5 * bookmark.importance)
        
        # Apply decay for stale bookmarks
        days_since_relevant = (now() - bookmark.last_relevant_at).days
        decay = max(0.5, 1.0 - (days_since_relevant * 0.05))  # Lose 5% per day, floor at 50%
        score *= decay
        
        bookmark_scores.append((bookmark, score))
    
    bookmark_scores.sort(key=lambda x: x[1], reverse=True)
    
    # === TIER 2: EXPAND TO RAW CHUNKS ===
    retrieved_chunk_ids = set()
    retrieved_chunks = []
    current_tokens = 0
    
    for bookmark, score in bookmark_scores:
        if score < 0.3:
            break
        
        for chunk_id in bookmark.source_chunk_ids:
            if chunk_id not in retrieved_chunk_ids:
                chunk = get_chunk(chunk_id, raw_chunks)
                chunk_tokens = count_tokens(chunk.content)
                
                if current_tokens + chunk_tokens <= context_budget * 0.7:
                    retrieved_chunks.append(chunk)
                    retrieved_chunk_ids.add(chunk_id)
                    current_tokens += chunk_tokens
                    
                    # Update bookmark relevance (it was useful)
                    bookmark.last_relevant_at = now()
    
    # === TIER 3: FILL WITH STANDARD RAG ===
    if current_tokens < context_budget * 0.7:
        remaining = context_budget - current_tokens
        additional = standard_similarity_search(
            query,
            raw_chunks,
            exclude_ids=retrieved_chunk_ids,
            budget=remaining
        )
        retrieved_chunks.extend(additional)
    
    # Sort chronologically
    retrieved_chunks.sort(key=lambda x: x.timestamp)
    
    return retrieved_chunks
```

**Changes from v1:**
- Removed fixed 80/20 budget split—now dynamically fills based on what's available
- Added decay for stale bookmarks
- Bookmarks update `last_relevant_at` when retrieved (prevents useful bookmarks from decaying)

---

## Algorithm 4: Refresh Flow

```python
def handle_refresh(
    conversation: List[ConversationChunk],
    bookmark_storage,
    raw_storage
):
    # Generate new bookmarks
    new_bookmarks = generate_bookmarks(conversation)
    
    # Merge with existing (keep old bookmarks that are still relevant)
    existing = bookmark_storage.get_all()
    merged = merge_bookmarks(existing, new_bookmarks)
    
    # Store
    bookmark_storage.replace_all(merged)
    raw_storage.upsert(conversation)
    
    return merged


def merge_bookmarks(existing: List[Bookmark], new: List[Bookmark]) -> List[Bookmark]:
    """
    Keep existing bookmarks that:
    - Were recently relevant (retrieved in past 7 days)
    - Cover content not in new bookmarks
    
    Replace with new bookmarks otherwise.
    """
    # Implementation: dedupe by source_chunk_ids overlap
    # Prefer new bookmarks when overlap > 50%
    # Keep old bookmarks for non-overlapping content if recently relevant
    pass
```

---

## Validation Strategy

**Before building the full system, validate the hypothesis.**

### Phase 1: Baseline Test (3 days)

Compare retrieval quality on 50 real long conversations:

```python
def baseline_test():
    for conversation in sample_conversations:
        test_queries = generate_test_queries(conversation)  # Or use real follow-up queries
        
        for query in test_queries:
            # Method A: Standard similarity
            result_a = similarity_search(query, conversation)
            
            # Method B: Recency-weighted similarity
            result_b = similarity_search(query, conversation, recency_weight=0.3)
            
            # Method C: This system (bookmarks first)
            result_c = hierarchical_retrieval(query, bookmarks, conversation)
            
            # Human eval: which retrieved the most relevant context?
            log_for_evaluation(query, result_a, result_b, result_c)
```

**Decision gate:** If Method B (recency-weighted) performs within 10% of Method C, don't build the bookmark layer. The complexity isn't worth it.

### Phase 2: Shadow Mode (2 weeks)

If Phase 1 passes, deploy bookmark generation without using it for retrieval:

```python
def shadow_mode():
    # Generate bookmarks on refresh
    bookmarks = generate_bookmarks(conversation)
    
    # Log them, but don't use for retrieval
    log_bookmarks_for_review(bookmarks)
    
    # Continue using standard RAG
    context = standard_similarity_search(query, conversation)
```

**Review questions:**
- Are the bookmarked moments actually the ones that matter?
- Does the LLM consistently identify the same patterns across runs?
- Are resolutions captured correctly, or are we still surfacing errors?

### Phase 3: A/B Test (2 weeks)

If Phase 2 looks good, run a proper A/B:

- Control: Standard RAG with recency weighting
- Treatment: Hierarchical bookmark retrieval

**Metrics:**
- User repetition rate (do they re-explain things less?)
- Explicit correction rate (do they say "as I mentioned" less?)
- Conversation length before abandonment
- Optional: user satisfaction survey

---

## Cost Analysis (Revised)

### Build Costs

| Component | Time | Notes |
|-----------|------|-------|
| Orchestration + trigger | 1 day | Length check, button UI |
| Bookmark generation prompt | 1 day | Prompt engineering, JSON parsing |
| Hierarchical retrieval | 2 days | Two-tier search, decay logic |
| Storage schema | 0.5 days | Add bookmark table to existing DB |
| Testing | 1.5 days | Integration tests, edge cases |
| **Total** | **~6 days** | Down from 2-4 weeks |

### Runtime Costs

| Operation | Cost | Frequency |
|-----------|------|-----------|
| Bookmark generation | ~$0.01-0.02 per refresh | Once per 50 messages or manual trigger |
| Retrieval | +10-20ms latency | Every query (two searches instead of one) |
| Storage | +15-20% | Bookmarks are small |

### Break-Even

Worth building if:
- Long conversations (50+ messages) are common in your product
- User repetition/frustration is a measurable problem
- Phase 1 baseline test shows bookmark retrieval outperforms recency weighting

Not worth building if:
- Most conversations are short
- Phase 1 shows recency weighting is "good enough"
- Engineering time is better spent elsewhere

---

## What Was Removed from v1

| Component | Reason for removal |
|-----------|-------------------|
| Per-message signal detection | LLM can notice patterns during refresh pass |
| Frustration/correction/convergence classifiers | Fragile, culturally variable, arbitrary thresholds |
| Complex refresh triggers (time, drift, signal density) | Length check is sufficient |
| Algorithm 4 (trigger logic) | Replaced with one-line threshold |
| Fixed 80/20 budget split | Dynamic allocation is more flexible |

---

## Remaining Risks

1. **LLM inconsistency.** The same conversation might produce different bookmarks on different runs. Mitigation: use low temperature, structured output format.

2. **Hallucinated bookmarks.** The LLM might summarize something that didn't happen. Mitigation: bookmarks point to source chunks; retrieval pulls the real content.

3. **Feedback loops.** If a bookmark is wrong but keeps getting retrieved, it persists. Mitigation: decay + user-triggered refresh.

4. **The core hypothesis might be wrong.** LLM-judged salience might not beat simple recency weighting. Mitigation: validate before building (Phase 1).

---

## Summary

This is a 1-week build that tests whether LLM-judged salience improves retrieval over simpler methods.

The key insight from the peer review: you don't need continuous signal detection. A single LLM pass at refresh time can identify frustration, corrections, and key moments just by reading the conversation.

Build the baseline test first. If recency weighting is good enough, stop. If not, you have a lightweight system that can be deployed incrementally.
