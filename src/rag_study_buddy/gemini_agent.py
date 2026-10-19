"""Terminal sample for the single-request Gemini RAG workflow."""

import sys

if __package__:
    from .gemini_rag import RagSession, describe_error, load_settings
else:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.rag_study_buddy.gemini_rag import RagSession, describe_error, load_settings


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        settings = load_settings()
        session = RagSession.prepare(settings, on_progress=lambda text: print(text, flush=True))
        for event in session.stream_answer("What is a subagent, and when should I use one?"):
            if event["type"] == "status":
                print(event["text"], flush=True)
            elif event["type"] == "complete":
                print("\nRetrieved passages:")
                answer = event["text"]
                for evidence in event["evidence"]:
                    print(f"\n[{evidence.number}] {evidence.source} (chunk {evidence.chunk_id})")
                    print(evidence.text[:600])
                    answer = answer.replace(f"(#{evidence.anchor})", f"({evidence.source})")
                print("\nAnswer:")
                print(answer)
    except Exception as error:
        print(describe_error(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
