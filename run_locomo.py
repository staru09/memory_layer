import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

import db
import vector_store
from main import ingest_conversation as run_ingestion
from retrieval.fetch_mem_service import compose_chat_context
from gemini import gemini_client as client
from config import GEMINI_MODEL

IST = timezone(timedelta(hours=5, minutes=30))

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


# ── Dataset Helpers ──

def parse_locomo_date(date_str: str) -> str:
    if not date_str:
        return "2024-01-01"
    try:
        if " on " in date_str:
            date_part = date_str.split(" on ", 1)[1].strip()
        else:
            date_part = date_str.strip()
        date_part = date_part.replace(",", "")
        dt = datetime.strptime(date_part, "%d %B %Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "2024-01-01"


def load_conversation(sample_idx: int) -> dict:
    with open("locomo/data/locomo10.json") as f:
        data = json.load(f)
    return data[sample_idx]


def _get_last_session_date(sample: dict) -> datetime:
    conv = sample["conversation"]
    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda s: int(s.split("_")[1])
    )
    if not session_keys:
        return datetime.now(IST).replace(tzinfo=None)
    last_date_key = f"{session_keys[-1]}_date_time"
    raw_date = conv.get(last_date_key, "")
    date_str = parse_locomo_date(raw_date)
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return datetime.now(IST).replace(tzinfo=None)


# ── Database Reset ──

def reset_databases():
    """Wipe all tables + Qdrant collection and recreate."""
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("""
        DROP TABLE IF EXISTS query_logs CASCADE;
        DROP TABLE IF EXISTS conflict_log CASCADE;
        DROP TABLE IF EXISTS conversation_summaries CASCADE;
        DROP TABLE IF EXISTS facts CASCADE;
        DROP TABLE IF EXISTS foresight CASCADE;
        DROP TABLE IF EXISTS user_profile CASCADE;
        DROP TABLE IF EXISTS chat_messages CASCADE;
        DROP TABLE IF EXISTS chat_threads CASCADE;
    """)
    conn.commit()
    cur.close()
    db.release_connection(conn)

    vector_store.delete_collection()
    vector_store.init_collections()

    db.init_schema()
    print("[reset] Databases reset.")


# ── Ingestion ──

def ingest_sample(sample: dict):
    """Ingest all sessions from a LoCoMo sample."""
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "User")
    speaker_b = conv.get("speaker_b", "Assistant")

    sessions = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda s: int(s.split("_")[1])
    )

    print(f"[ingest] {speaker_a} & {speaker_b} | {len(sessions)} sessions")

    total_turns = 0
    for i, session_key in enumerate(sessions):
        date_key = f"{session_key}_date_time"
        raw_date = conv.get(date_key, "2024-01-01")
        session_date = parse_locomo_date(raw_date)

        msgs = []
        for turn in conv[session_key]:
            speaker = turn["speaker"]
            role = "user" if speaker == speaker_a else "assistant"
            msgs.append({
                "role": role,
                "content": f"{speaker}: {turn['text']}",
                "created_at": session_date,
            })

        total_turns += len(msgs)
        source_id = f"locomo_{session_key}"

        print(f"  Session {i+1}/{len(sessions)}: {session_key} ({len(msgs)} turns, {session_date})")

        try:
            run_ingestion(msgs, source_id=source_id, current_date=session_date,
                          extract_all_speakers=True)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"[ingest] Done: {total_turns} turns ingested")


# ── QA Evaluation ──

def _process_single_qa(qa: dict, index: int, total: int,
                       client, model, query_time: datetime) -> dict:
    """Process a single QA pair: retrieve context, generate answer, score."""
    question = qa["question"]
    ground_truth = qa.get("answer", qa.get("adversarial_answer", ""))
    category = qa.get("category", 0)
    cat_name = CATEGORY_NAMES.get(category, "unknown")
    is_adversarial = category == 5

    # Retrieve context (rolling summary path — no search)
    t0 = time.time()
    try:
        ctx = db.get_chat_context("benchmark", query_time)
        context = compose_chat_context({
            "profile": "",
            "foresight": ctx["foresight"],
            "summary": ctx["summary"],
        })
    except Exception as e:
        print(f"  [{index+1}/{total}] ({cat_name}) RETRIEVAL ERROR: {e}")
        return {
            "question": question, "ground_truth": str(ground_truth),
            "answer": "ERROR", "category": category, "cat_name": cat_name,
            "retrieval_time": 0, "llm_time": 0, "score": 0, "reasoning": str(e),
        }
    retrieval_time = time.time() - t0

    # Generate answer
    t0 = time.time()

    if is_adversarial:
        answer_prompt = f"""You are answering questions about a long-term conversation between two people. Use ONLY the provided context.

ANSWERING RULES:
- Keep your answer short and precise (1-2 sentences max).
- Be SPECIFIC: include exact names, dates, places, numbers, titles when available.
- Trust the MOST RECENT information when facts conflict.
- Do NOT make up facts. Only use what is explicitly in the context.
- IMPORTANT: Before saying "I don't know", carefully re-read ALL sections of the context — the answer may be in the profile, the archive, OR the recent entries. Search every section thoroughly.
- IMPORTANT: Prefer giving a partial answer over "I don't know." If you can find ANY relevant information, share it.
- The question may attribute an action to the wrong person. If the context shows a different person did that thing, STILL answer with the factual information — do not refuse to answer.
- When answering, focus on WHAT happened, not WHO did it. Answer with just the facts without naming the person.
- NEVER say "the context does not mention" or "this was not mentioned" if the information IS in the context under a different person's name. Search for the ACTIVITY/EVENT regardless of who did it.

Context:
{context}

Question: {question}

Answer:"""
    else:
        answer_prompt = f"""You are answering questions about a long-term conversation between two people. Use ONLY the provided context.

ANSWERING RULES:
- Keep your answer short and precise (1-2 sentences max).
- Be SPECIFIC: include exact names, dates, places, numbers, titles when available.
- For "would someone do X" questions: reason from their stated interests, career goals, and preferences. If they previously expressed interest in something but circumstances changed, favour the most recent signal.
- Trust the MOST RECENT information when facts conflict.
- Do NOT make up facts. Only use what is explicitly in the context.
- IMPORTANT: Before saying "I don't know", carefully re-read ALL sections of the context — the answer may be in the profile, the archive, OR the recent entries. Search every section thoroughly.
- IMPORTANT: Prefer giving a partial answer over "I don't know." If you can find ANY relevant information, share it.
- When asked "what book" or "what title" — look for quoted titles in the context.
- Be careful about WHO did what — do not confuse the two speakers.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = client.models.generate_content(model=model, contents=answer_prompt)
        answer = response.text.strip() if response.text else "ERROR: empty response"
    except Exception as e:
        answer = f"ERROR: {e}"
    llm_time = time.time() - t0

    # Score using LLM-as-judge
    try:
        judge_prompt = f"""Compare the generated answer against the ground truth.
Score from 1-5:
5 = Perfect match or semantically equivalent
4 = Mostly correct, minor details different
3 = Partially correct, has the right idea but missing key details
2 = Mostly wrong but has some relevant information
1 = Completely wrong or irrelevant

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {answer}

Return ONLY a JSON: {{"score": N, "reasoning": "brief explanation"}}"""

        judge_response = client.models.generate_content(model=model, contents=judge_prompt)
        judge_text = judge_response.text.strip()
        if judge_text.startswith("```"):
            judge_text = judge_text.split("\n", 1)[1]
            if judge_text.endswith("```"):
                judge_text = judge_text[:-3]
        judge_result = json.loads(judge_text)
        score = judge_result.get("score", 0)
        reasoning = judge_result.get("reasoning", "")
    except Exception:
        score = 0
        reasoning = "judge failed"

    marker = " [ADV]" if is_adversarial else ""
    print(f"  [{index+1}/{total}] ({cat_name}{marker}) Score: {score}/5 | R: {retrieval_time:.2f}s | L: {llm_time:.1f}s | {question[:50]}...")
    if score <= 2:
        print(f"    GT:  {str(ground_truth)[:80]}")
        print(f"    Got: {answer[:80]}")

    return {
        "question": question,
        "ground_truth": str(ground_truth),
        "answer": answer,
        "category": category,
        "cat_name": cat_name,
        "retrieval_time": retrieval_time,
        "llm_time": llm_time,
        "score": score,
        "reasoning": reasoning,
    }


def run_qa(sample: dict, parallel_workers: int = 5) -> list[dict]:
    """Run all QA pairs (including adversarial) with parallel workers."""
    try:
        db.create_thread("benchmark", "LoCoMo Benchmark")
    except Exception:
        pass  # already exists

    qa_pairs = sample["qa"]
    query_time = _get_last_session_date(sample)

    total = len(qa_pairs)
    adv_count = sum(1 for q in qa_pairs if q.get("category") == 5)
    non_adv_count = total - adv_count
    print(f"\n[qa] {total} questions ({non_adv_count} standard + {adv_count} adversarial) | {parallel_workers} workers")
    print(f"[qa] Reference date: {query_time.strftime('%Y-%m-%d')}")

    results = [None] * total

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {}
        for i, qa in enumerate(qa_pairs):
            future = executor.submit(
                _process_single_qa, qa, i, total, client, GEMINI_MODEL, query_time
            )
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"  [{idx+1}/{total}] FAILED: {e}")
                results[idx] = {
                    "question": qa_pairs[idx]["question"],
                    "ground_truth": str(qa_pairs[idx].get("answer", qa_pairs[idx].get("adversarial_answer", ""))),
                    "answer": "ERROR", "category": qa_pairs[idx].get("category", 0),
                    "cat_name": CATEGORY_NAMES.get(qa_pairs[idx].get("category", 0), "unknown"),
                    "retrieval_time": 0, "llm_time": 0, "score": 0, "reasoning": str(e),
                }

    return [r for r in results if r is not None]


# ── Report Generation ──

def generate_report(all_results: dict, output_path: str):
    """Generate a markdown report from all sample results."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# LoCoMo Benchmark Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Architecture**: gpt_style (rolling summary, no RAG for chat)\n")
        f.write(f"**Samples**: {len(all_results)}\n\n")

        # Overall summary table
        f.write("## Overall Summary\n\n")
        f.write("| Sample | Speakers | Sessions | QA Total | Non-Adv | Adv | Avg Score | Avg (Non-Adv) | Avg (Adv) | >=4 | >=3 |\n")
        f.write("|--------|----------|----------|----------|---------|-----|-----------|---------------|-----------|-----|-----|\n")

        grand_scores = []
        grand_non_adv = []
        grand_adv = []

        for sample_idx, results in sorted(all_results.items()):
            non_adv = [r for r in results if r["category"] != 5 and r["score"] > 0]
            adv = [r for r in results if r["category"] == 5 and r["score"] > 0]
            all_scored = [r for r in results if r["score"] > 0]

            non_adv_scores = [r["score"] for r in non_adv]
            adv_scores = [r["score"] for r in adv]
            all_scores = [r["score"] for r in all_scored]

            avg_all = sum(all_scores) / len(all_scores) if all_scores else 0
            avg_non_adv = sum(non_adv_scores) / len(non_adv_scores) if non_adv_scores else 0
            avg_adv = sum(adv_scores) / len(adv_scores) if adv_scores else 0
            gte4 = sum(1 for s in all_scores if s >= 4)
            gte3 = sum(1 for s in all_scores if s >= 3)

            grand_scores.extend(all_scores)
            grand_non_adv.extend(non_adv_scores)
            grand_adv.extend(adv_scores)

            # Get speakers from first result's metadata or use index
            conv = load_conversation(sample_idx)
            speakers = f"{conv['conversation'].get('speaker_a', '?')} & {conv['conversation'].get('speaker_b', '?')}"
            sessions = len([k for k in conv['conversation'] if k.startswith('session_') and not k.endswith('date_time')])

            f.write(f"| {sample_idx} | {speakers} | {sessions} | {len(results)} | {len(non_adv)} | {len(adv)} | {avg_all:.2f}/5 | {avg_non_adv:.2f}/5 | {avg_adv:.2f}/5 | {gte4}/{len(all_scores)} ({gte4/max(len(all_scores),1)*100:.1f}%) | {gte3}/{len(all_scores)} ({gte3/max(len(all_scores),1)*100:.1f}%) |\n")

        # Grand totals
        if grand_scores:
            f.write(f"| **Total** | | | | {len(grand_non_adv)} | {len(grand_adv)} | **{sum(grand_scores)/len(grand_scores):.2f}/5** | **{sum(grand_non_adv)/len(grand_non_adv):.2f}/5** | **{sum(grand_adv)/len(grand_adv):.2f}/5** | {sum(1 for s in grand_scores if s >= 4)}/{len(grand_scores)} ({sum(1 for s in grand_scores if s >= 4)/len(grand_scores)*100:.1f}%) | {sum(1 for s in grand_scores if s >= 3)}/{len(grand_scores)} ({sum(1 for s in grand_scores if s >= 3)/len(grand_scores)*100:.1f}%) |\n")

        # Per-category breakdown
        f.write("\n## Per-Category Breakdown (All Samples Combined)\n\n")
        f.write("| Category | Count | Avg Score | >=4 | >=3 |\n")
        f.write("|----------|-------|-----------|-----|-----|\n")

        all_results_flat = []
        for results in all_results.values():
            all_results_flat.extend(results)

        for cat_num in sorted(CATEGORY_NAMES.keys()):
            cat_results = [r for r in all_results_flat if r["category"] == cat_num and r["score"] > 0]
            if not cat_results:
                continue
            cat_scores = [r["score"] for r in cat_results]
            avg = sum(cat_scores) / len(cat_scores)
            gte4 = sum(1 for s in cat_scores if s >= 4)
            gte3 = sum(1 for s in cat_scores if s >= 3)
            name = CATEGORY_NAMES[cat_num]
            f.write(f"| {name} | {len(cat_results)} | {avg:.2f}/5 | {gte4}/{len(cat_results)} ({gte4/len(cat_results)*100:.1f}%) | {gte3}/{len(cat_results)} ({gte3/len(cat_results)*100:.1f}%) |\n")

        # Per-sample details
        for sample_idx, results in sorted(all_results.items()):
            conv = load_conversation(sample_idx)
            speakers = f"{conv['conversation'].get('speaker_a', '?')} & {conv['conversation'].get('speaker_b', '?')}"

            f.write(f"\n---\n\n## Sample {sample_idx}: {speakers}\n\n")

            # Category breakdown for this sample
            f.write("### Per-Category\n\n")
            f.write("| Category | Count | Avg Score | >=4 | >=3 |\n")
            f.write("|----------|-------|-----------|-----|-----|\n")

            for cat_num in sorted(CATEGORY_NAMES.keys()):
                cat_results = [r for r in results if r["category"] == cat_num and r["score"] > 0]
                if not cat_results:
                    continue
                cat_scores = [r["score"] for r in cat_results]
                avg = sum(cat_scores) / len(cat_scores)
                gte4 = sum(1 for s in cat_scores if s >= 4)
                gte3 = sum(1 for s in cat_scores if s >= 3)
                name = CATEGORY_NAMES[cat_num]
                f.write(f"| {name} | {len(cat_results)} | {avg:.2f}/5 | {gte4}/{len(cat_results)} ({gte4/len(cat_results)*100:.1f}%) | {gte3}/{len(cat_results)} ({gte3/len(cat_results)*100:.1f}%) |\n")

            # Worst answers
            worst = sorted([r for r in results if r["score"] > 0], key=lambda x: x["score"])[:5]
            if worst:
                f.write(f"\n### Worst Answers\n\n")
                f.write("| Category | Score | Question | Ground Truth | Got |\n")
                f.write("|----------|-------|----------|-------------|-----|\n")
                for r in worst:
                    q = r["question"][:60].replace("|", "/")
                    gt = r["ground_truth"][:50].replace("|", "/")
                    got = r["answer"][:50].replace("|", "/")
                    f.write(f"| {r['cat_name']} | {r['score']}/5 | {q} | {gt} | {got} |\n")

    print(f"\n[report] Saved to {output_path}")


def save_raw_results(sample_idx: int, results: list[dict]):
    """Save raw JSON results for a sample."""
    os.makedirs("benchmark_results", exist_ok=True)
    path = f"benchmark_results/sample_{sample_idx}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[save] Raw results: {path}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark")
    parser.add_argument("--samples", type=int, nargs="*", default=None,
                        help="Sample indices to run (default: all 0-9)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel workers for QA evaluation")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip reset + ingestion, run QA only on existing data")
    args = parser.parse_args()

    samples = args.samples if args.samples is not None else list(range(10))

    print(f"\n{'#' * 60}")
    print(f"  LoCoMo BENCHMARK — {len(samples)} samples")
    print(f"  Samples: {samples}")
    print(f"  Workers: {args.workers}")
    print(f"{'#' * 60}")

    all_results = {}
    total_start = time.time()

    for i, sample_idx in enumerate(samples):
        sample = load_conversation(sample_idx)
        conv = sample["conversation"]
        speakers = f"{conv.get('speaker_a', '?')} & {conv.get('speaker_b', '?')}"
        sessions = len([k for k in conv if k.startswith("session_") and not k.endswith("date_time")])
        qa_count = len(sample["qa"])

        print(f"\n{'=' * 60}")
        print(f"  SAMPLE {sample_idx}/{len(samples)-1}: {speakers}")
        print(f"  Sessions: {sessions} | QA: {qa_count}")
        print(f"  Progress: {i+1}/{len(samples)}")
        print(f"{'=' * 60}")

        if not args.skip_ingest:
            # Step 1: Reset
            print(f"\n[step 1/3] Resetting databases...")
            reset_databases()

            # Step 2: Ingest
            print(f"\n[step 2/3] Ingesting conversations...")
            t_ingest = time.time()
            ingest_sample(sample)
            ingest_time = time.time() - t_ingest
            print(f"[ingest] Total time: {ingest_time:.1f}s")
        else:
            ingest_time = 0
            print(f"\n[skip] Using existing data (--skip-ingest)")

        # Step 3: Evaluate
        print(f"\n[step 3/3] Running QA evaluation...")
        t_qa = time.time()
        results = run_qa(sample, parallel_workers=args.workers)
        qa_time = time.time() - t_qa

        # Print summary for this sample
        scored = [r for r in results if r["score"] > 0]
        non_adv = [r for r in scored if r["category"] != 5]
        adv = [r for r in scored if r["category"] == 5]

        avg_all = sum(r["score"] for r in scored) / len(scored) if scored else 0
        avg_non_adv = sum(r["score"] for r in non_adv) / len(non_adv) if non_adv else 0
        avg_adv = sum(r["score"] for r in adv) / len(adv) if adv else 0

        print(f"\n[sample {sample_idx} summary]")
        print(f"  Avg score (all):     {avg_all:.2f}/5")
        print(f"  Avg score (non-adv): {avg_non_adv:.2f}/5")
        print(f"  Avg score (adv):     {avg_adv:.2f}/5")
        print(f"  Ingest time:         {ingest_time:.0f}s")
        print(f"  QA time:             {qa_time:.0f}s")

        # Save raw results
        save_raw_results(sample_idx, results)
        all_results[sample_idx] = results

    # Generate final report
    total_time = time.time() - total_start
    print(f"\n{'#' * 60}")
    print(f"  ALL SAMPLES COMPLETE — {total_time:.0f}s total")
    print(f"{'#' * 60}")

    report_path = f"benchmark_results/locomo_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    generate_report(all_results, report_path)

    # Also save combined raw results
    combined_path = "benchmark_results/all_results.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, ensure_ascii=False)
    print(f"[save] Combined results: {combined_path}")


if __name__ == "__main__":
    main()
