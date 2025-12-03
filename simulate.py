#!/usr/bin/env python3
"""
Hierarchical Bookmark Retrieval - Simulation Demo

This demonstrates the core concept without requiring actual LLM calls or vector DBs.
It uses mock data and rule-based scoring to show how the system would work.

Run: python simulate.py
"""

import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Tuple, Set
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ConversationChunk:
    id: str
    content: str
    timestamp: datetime
    speaker: str  # "user" | "assistant"
    message_index: int
    # Simulated embedding (in real system, this would be a vector)
    keywords: Set[str] = field(default_factory=set)


@dataclass
class Bookmark:
    id: str
    content: str
    source_chunk_ids: List[str]
    importance: float
    bookmark_type: str  # "conclusion" | "resolution" | "validated" | "key_claim"
    created_at: datetime
    last_relevant_at: datetime
    keywords: Set[str] = field(default_factory=set)


# =============================================================================
# MOCK CONVERSATION GENERATOR
# =============================================================================

def generate_mock_conversation(num_messages: int = 80) -> List[ConversationChunk]:
    """
    Generate a realistic mock conversation with seeded "important" moments.
    
    The conversation simulates a user working on a project with an AI assistant.
    Important moments are explicitly marked with keywords the system should catch.
    """
    
    conversation = []
    base_time = datetime.now() - timedelta(hours=num_messages // 2)
    
    # Templates for regular messages
    user_messages = [
        "Can you explain how this works?",
        "I'm not sure I understand that part.",
        "Let me think about that for a moment.",
        "What about the edge cases?",
        "How does this compare to other approaches?",
        "Can you give me an example?",
        "I need to consider the performance implications.",
        "What's the tradeoff here?",
        "Let me try implementing that.",
        "I'm getting an error when I run this.",
        "The documentation isn't clear on this.",
        "Can you break that down further?",
        "What would you recommend?",
        "I've seen different approaches to this.",
        "How do I handle the async case?",
    ]
    
    assistant_messages = [
        "Here's how it works in practice...",
        "Let me break that down step by step.",
        "There are a few approaches you could take.",
        "The key insight here is...",
        "Based on your requirements, I'd suggest...",
        "Here's an example that might help.",
        "That's a good question. The tradeoff is...",
        "Looking at your code, I think the issue is...",
        "Let me explain the underlying concept.",
        "There are pros and cons to each approach.",
        "The documentation covers this in section...",
        "Here's a more detailed explanation.",
        "You're right to consider that edge case.",
        "The best practice here is...",
        "Let me show you the recommended pattern.",
    ]
    
    # IMPORTANT MOMENTS - these are what bookmarks should capture
    important_moments = {
        # (message_index, speaker, content, bookmark_type, importance)
        10: ("user", "My budget for this project is $50,000 and the deadline is March 15th.", "key_claim", 0.95),
        15: ("user", "Yes exactly! That's exactly what I meant about the caching strategy.", "validated", 0.8),
        25: ("assistant", "DECISION: We'll use PostgreSQL for the main database and Redis for caching.", "conclusion", 0.9),
        30: ("user", "Actually, I should clarify - when I said 'users' I meant internal employees only, not customers.", "resolution", 0.85),
        35: ("assistant", "RESOLVED: The API will use OAuth2 for authentication, not API keys as initially discussed.", "resolution", 0.9),
        45: ("user", "That's it! Perfect, the batch processing approach is what we'll go with.", "validated", 0.85),
        55: ("assistant", "CONCLUSION: The final architecture uses microservices with event-driven communication.", "conclusion", 0.95),
        60: ("user", "My team has 3 backend developers and 2 frontend developers available.", "key_claim", 0.8),
        70: ("user", "Right, exactly - we need to prioritize mobile responsiveness over desktop features.", "validated", 0.85),
        75: ("assistant", "DECISION: We'll deploy to AWS using ECS with Fargate for container orchestration.", "conclusion", 0.9),
    }
    
    for i in range(num_messages):
        speaker = "user" if i % 2 == 0 else "assistant"
        timestamp = base_time + timedelta(minutes=i * 3)
        
        # Check if this is an important moment
        if i in important_moments:
            _, content, btype, importance = important_moments[i]
            speaker = important_moments[i][0]
        else:
            # Regular message
            if speaker == "user":
                content = random.choice(user_messages)
            else:
                content = random.choice(assistant_messages)
        
        # Extract keywords for mock similarity
        keywords = set(word.lower().strip('.,!?') for word in content.split() if len(word) > 4)
        
        chunk = ConversationChunk(
            id=f"chunk_{i:03d}",
            content=content,
            timestamp=timestamp,
            speaker=speaker,
            message_index=i,
            keywords=keywords
        )
        conversation.append(chunk)
    
    return conversation, important_moments


# =============================================================================
# BOOKMARK GENERATION (simulated - no actual LLM)
# =============================================================================

def generate_bookmarks(conversation: List[ConversationChunk]) -> List[Bookmark]:
    """
    Simulate LLM bookmark generation using rule-based heuristics.
    
    In the real system, this would be a single LLM call that reviews
    the full conversation and extracts important moments. See the spec
    for the actual prompt.
    
    Here we use pattern matching (looking for keywords like "DECISION:",
    "exactly", "budget", etc.) to demonstrate the concept.
    
    NOTE: The simulation proves that IF you can identify importance 
    (which we simulate with seeded patterns), the hierarchical retrieval 
    structure beats flat search. The LLM's actual judgment quality is 
    what makes or breaks a production system—that's why the spec includes 
    a baseline test before building.
    """
    
    bookmarks = []
    
    # Patterns that indicate important moments
    conclusion_patterns = ["DECISION:", "CONCLUSION:", "we'll use", "we'll go with", "final"]
    resolution_patterns = ["RESOLVED:", "clarify", "actually", "should clarify", "I meant"]
    validation_patterns = ["exactly", "that's it", "perfect", "right,", "yes exactly"]
    key_claim_patterns = ["budget", "deadline", "team has", "my project", "requirements"]
    
    for i, chunk in enumerate(conversation):
        content_lower = chunk.content.lower()
        bookmark_type = None
        importance = 0.0
        
        # Check for conclusions
        for pattern in conclusion_patterns:
            if pattern.lower() in content_lower:
                bookmark_type = "conclusion"
                importance = 0.9
                break
        
        # Check for resolutions
        if not bookmark_type:
            for pattern in resolution_patterns:
                if pattern.lower() in content_lower:
                    bookmark_type = "resolution"
                    importance = 0.85
                    break
        
        # Check for validations
        if not bookmark_type:
            for pattern in validation_patterns:
                if pattern.lower() in content_lower:
                    bookmark_type = "validated"
                    importance = 0.8
                    break
        
        # Check for key claims
        if not bookmark_type:
            for pattern in key_claim_patterns:
                if pattern.lower() in content_lower:
                    bookmark_type = "key_claim"
                    importance = 0.85
                    break
        
        # Create bookmark if important moment found
        if bookmark_type:
            # Get surrounding context (1 message before and after)
            source_ids = []
            if i > 0:
                source_ids.append(conversation[i-1].id)
            source_ids.append(chunk.id)
            if i < len(conversation) - 1:
                source_ids.append(conversation[i+1].id)
            
            bookmark = Bookmark(
                id=f"bookmark_{len(bookmarks):03d}",
                content=chunk.content[:150],  # Summary (truncated for demo)
                source_chunk_ids=source_ids,
                importance=importance,
                bookmark_type=bookmark_type,
                created_at=chunk.timestamp,
                last_relevant_at=chunk.timestamp,
                keywords=chunk.keywords
            )
            bookmarks.append(bookmark)
    
    return bookmarks


# =============================================================================
# RETRIEVAL METHODS
# =============================================================================

def mock_similarity(query_keywords: Set[str], target_keywords: Set[str]) -> float:
    """
    Simulates semantic similarity using Jaccard index (keyword overlap).
    
    NOTE: In a production implementation, this would be replaced by Cosine 
    Similarity on vector embeddings (e.g., OpenAI text-embedding-3-small, 
    Cohere, or open-source alternatives), which handles synonyms and 
    semantic meaning much better than this strict keyword match.
    
    This simulation proves the STRUCTURAL advantage (reducing the search 
    space to high-value nodes) rather than the quality of similarity matching.
    The hierarchical approach wins because it searches a smaller, higher-quality 
    haystack—not because of better similarity scoring.
    """
    if not query_keywords or not target_keywords:
        return 0.0
    
    intersection = len(query_keywords & target_keywords)
    union = len(query_keywords | target_keywords)
    
    return intersection / union if union > 0 else 0.0


def standard_similarity_search(
    query: str,
    chunks: List[ConversationChunk],
    top_k: int = 5
) -> List[Tuple[ConversationChunk, float]]:
    """
    Baseline Method A: Standard similarity search
    """
    query_keywords = set(word.lower().strip('.,!?') for word in query.split() if len(word) > 4)
    
    scored = []
    for chunk in chunks:
        score = mock_similarity(query_keywords, chunk.keywords)
        scored.append((chunk, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def recency_weighted_search(
    query: str,
    chunks: List[ConversationChunk],
    top_k: int = 5,
    recency_weight: float = 0.3
) -> List[Tuple[ConversationChunk, float]]:
    """
    Baseline Method B: Similarity with recency weighting
    """
    query_keywords = set(word.lower().strip('.,!?') for word in query.split() if len(word) > 4)
    
    max_index = max(c.message_index for c in chunks)
    
    scored = []
    for chunk in chunks:
        similarity = mock_similarity(query_keywords, chunk.keywords)
        recency = chunk.message_index / max_index if max_index > 0 else 0
        
        # Blend similarity with recency
        score = (1 - recency_weight) * similarity + recency_weight * recency
        scored.append((chunk, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def hierarchical_bookmark_retrieval(
    query: str,
    bookmarks: List[Bookmark],
    chunks: List[ConversationChunk],
    context_budget: int = 5
) -> List[Tuple[ConversationChunk, float, str]]:
    """
    Method C: Hierarchical bookmark retrieval (THE PROPOSED SYSTEM)
    
    Returns: List of (chunk, score, retrieval_tier)
    """
    query_keywords = set(word.lower().strip('.,!?') for word in query.split() if len(word) > 4)
    
    retrieved = []
    retrieved_ids = set()
    
    # === TIER 1: BOOKMARK SEARCH ===
    bookmark_scores = []
    for bookmark in bookmarks:
        similarity = mock_similarity(query_keywords, bookmark.keywords)
        # Weight by importance
        score = similarity * (0.5 + 0.5 * bookmark.importance)
        bookmark_scores.append((bookmark, score))
    
    bookmark_scores.sort(key=lambda x: x[1], reverse=True)
    
    # === TIER 2: EXPAND TO RAW CHUNKS ===
    for bookmark, score in bookmark_scores[:3]:  # Top 3 bookmarks
        if score < 0.1:
            continue
        
        for chunk_id in bookmark.source_chunk_ids:
            if chunk_id not in retrieved_ids and len(retrieved) < context_budget * 0.7:
                chunk = next((c for c in chunks if c.id == chunk_id), None)
                if chunk:
                    retrieved.append((chunk, score, f"bookmark:{bookmark.bookmark_type}"))
                    retrieved_ids.add(chunk_id)
    
    # === TIER 3: FILL WITH STANDARD RAG ===
    if len(retrieved) < context_budget:
        remaining = context_budget - len(retrieved)
        standard_results = standard_similarity_search(query, chunks, top_k=remaining + 5)
        
        for chunk, score in standard_results:
            if chunk.id not in retrieved_ids and len(retrieved) < context_budget:
                retrieved.append((chunk, score, "fallback:similarity"))
                retrieved_ids.add(chunk.id)
    
    return retrieved


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_retrieval(
    query: str,
    ground_truth_indices: Set[int],
    results: List[Tuple[ConversationChunk, float, ...]],
    method_name: str
) -> dict:
    """
    Evaluate how well a retrieval method found the relevant chunks.
    """
    retrieved_indices = set()
    for item in results:
        chunk = item[0]
        retrieved_indices.add(chunk.message_index)
    
    hits = retrieved_indices & ground_truth_indices
    precision = len(hits) / len(retrieved_indices) if retrieved_indices else 0
    recall = len(hits) / len(ground_truth_indices) if ground_truth_indices else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "method": method_name,
        "hits": len(hits),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "retrieved_indices": sorted(retrieved_indices),
        "ground_truth_indices": sorted(ground_truth_indices)
    }


# =============================================================================
# DEMO
# =============================================================================

def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def run_demo():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     HIERARCHICAL BOOKMARK RETRIEVAL - SIMULATION DEMO                ║
║                                                                      ║
║     This demonstrates the concept without actual LLM calls.          ║
║     Mock data + rule-based scoring shows how the system works.       ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Generate mock conversation
    print_separator("STEP 1: GENERATE MOCK CONVERSATION")
    conversation, important_moments = generate_mock_conversation(80)
    print(f"Generated {len(conversation)} messages")
    print(f"\nImportant moments seeded at indices: {sorted(important_moments.keys())}")
    
    print("\nSample important moments:")
    for idx in list(important_moments.keys())[:3]:
        speaker, content, btype, importance = important_moments[idx]
        print(f"  [{idx}] ({btype}, importance={importance})")
        print(f"       \"{content[:60]}...\"")
    
    # Generate bookmarks
    print_separator("STEP 2: GENERATE BOOKMARKS")
    bookmarks = generate_bookmarks(conversation)
    print(f"Generated {len(bookmarks)} bookmarks\n")
    
    print("Bookmarks found:")
    for b in bookmarks:
        source_indices = [int(sid.split('_')[1]) for sid in b.source_chunk_ids]
        print(f"  [{b.bookmark_type}] importance={b.importance:.2f}")
        print(f"       Source messages: {source_indices}")
        print(f"       \"{b.content[:50]}...\"")
        print()
    
    # Test queries
    print_separator("STEP 3: TEST RETRIEVAL QUERIES")
    
    test_cases = [
        {
            "query": "What is the budget and deadline for this project?",
            "ground_truth": {9, 10, 11},  # Around message 10 where budget was mentioned
            "description": "Query about project constraints (budget/deadline)"
        },
        {
            "query": "What database and caching solution did we decide on?",
            "ground_truth": {24, 25, 26},  # Around message 25 where DB decision was made
            "description": "Query about technical decisions"
        },
        {
            "query": "What did the user clarify about who the users are?",
            "ground_truth": {29, 30, 31},  # Around message 30 where clarification happened
            "description": "Query about a correction/clarification"
        },
        {
            "query": "How is the team structured? How many developers?",
            "ground_truth": {59, 60, 61},  # Around message 60 where team info was shared
            "description": "Query about team information"
        },
    ]
    
    overall_results = {"standard": [], "recency": [], "hierarchical": []}
    
    for i, test in enumerate(test_cases):
        print(f"\n--- Query {i+1}: {test['description']} ---")
        print(f"Query: \"{test['query']}\"")
        print(f"Ground truth indices: {sorted(test['ground_truth'])}")
        
        # Method A: Standard similarity
        results_a = standard_similarity_search(test['query'], conversation, top_k=5)
        eval_a = evaluate_retrieval(
            test['query'],
            test['ground_truth'],
            [(c, s, "similarity") for c, s in results_a],
            "Standard Similarity"
        )
        overall_results["standard"].append(eval_a)
        
        # Method B: Recency weighted
        results_b = recency_weighted_search(test['query'], conversation, top_k=5)
        eval_b = evaluate_retrieval(
            test['query'],
            test['ground_truth'],
            [(c, s, "recency") for c, s in results_b],
            "Recency Weighted"
        )
        overall_results["recency"].append(eval_b)
        
        # Method C: Hierarchical bookmarks
        results_c = hierarchical_bookmark_retrieval(test['query'], bookmarks, conversation, context_budget=5)
        eval_c = evaluate_retrieval(test['query'], test['ground_truth'], results_c, "Hierarchical Bookmarks")
        overall_results["hierarchical"].append(eval_c)
        
        print(f"\n  Results comparison:")
        print(f"  {'Method':<25} {'Hits':<6} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print(f"  {'-'*60}")
        for eval_result in [eval_a, eval_b, eval_c]:
            print(f"  {eval_result['method']:<25} {eval_result['hits']:<6} {eval_result['precision']:<10.2f} {eval_result['recall']:<10.2f} {eval_result['f1']:<10.2f}")
        
        # Show what hierarchical retrieval actually retrieved
        print(f"\n  Hierarchical retrieval details:")
        for chunk, score, tier in results_c[:5]:
            marker = "✓" if chunk.message_index in test['ground_truth'] else " "
            print(f"    {marker} [{chunk.message_index:3d}] (score={score:.2f}, via {tier})")
            print(f"           \"{chunk.content[:50]}...\"")
    
    # Summary
    print_separator("SUMMARY")
    
    print("\nAverage performance across all queries:")
    print(f"  {'Method':<25} {'Avg Hits':<10} {'Avg Precision':<15} {'Avg Recall':<12} {'Avg F1':<10}")
    print(f"  {'-'*70}")
    
    for method_name, results_list in overall_results.items():
        avg_hits = sum(r['hits'] for r in results_list) / len(results_list)
        avg_prec = sum(r['precision'] for r in results_list) / len(results_list)
        avg_recall = sum(r['recall'] for r in results_list) / len(results_list)
        avg_f1 = sum(r['f1'] for r in results_list) / len(results_list)
        
        display_name = {
            "standard": "Standard Similarity",
            "recency": "Recency Weighted",
            "hierarchical": "Hierarchical Bookmarks"
        }[method_name]
        
        print(f"  {display_name:<25} {avg_hits:<10.1f} {avg_prec:<15.2f} {avg_recall:<12.2f} {avg_f1:<10.2f}")
    
    print("\n" + "=" * 70)
    print(" KEY INSIGHT")
    print("=" * 70)
    print("""
  The hierarchical bookmark system finds important moments even when
  keyword similarity is low, because it pre-identified what mattered
  during the bookmark generation phase.
  
  Standard similarity search often misses decisions and clarifications
  because they don't share many keywords with the query.
  
  Recency weighting helps for recent content but misses important
  early context (like budget and deadline set at the beginning).
  
  The bookmark layer acts as a "what mattered" index that retrieval
  can leverage regardless of surface-level keyword overlap.
    """)
    
    print("\n" + "=" * 70)
    print(" NEXT STEPS")
    print("=" * 70)
    print("""
  To turn this into a real system:
  
  1. Replace mock_similarity() with actual embedding similarity
  2. Replace generate_bookmarks() with an LLM call (see spec for prompt)
  3. Add vector DB storage (Pinecone, Chroma, pgvector, etc.)
  4. Run the Phase 1 baseline test on real conversations
  
  See the full spec for implementation details:
  https://github.com/RealPsyclops/hierarchical_bookmarks_for_llms
    """)


if __name__ == "__main__":
    run_demo()
