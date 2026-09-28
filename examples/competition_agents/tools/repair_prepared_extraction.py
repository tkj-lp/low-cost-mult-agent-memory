"""仅基于原始来源修正离线抽取；不读取问题、答案或支撑标签。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

from tools.build_prepared_knowledge import PROMPT, read_env
from competition_agents.handoff import json_object
from competition_agents.llm.audited_chat import AuditedChat
from hypergraph_memory_method.prepared_knowledge import digest, validate_extraction

REVIEW = '''You are checking a knowledge extraction against its original document, not answering questions.
Return EXACTLY {"facts":[{"subject":"...","relation":"...","object":"...","polarity":"positive","qualifiers":[],"sentences":[0]}]} with ALL facts, not only changed facts.
The only top-level key is facts. Do NOT echo source, previous_extraction or validation_error.
Correct omitted claims and unsupported values. Copy subjects, objects and qualifier values as exact
contiguous source substrings; relation labels may be normalized. Do not add quotation delimiters to names.
For a conjunction such as "ambassador to A and to B", use relation="ambassador to", object="B" and
carry other necessary qualifications. Intransitive events like "closed" may have object="".
Resolve he/she/they/it when the antecedent is explicit, otherwise retain a source-local reference.
polarity=negative means grammatical denial ONLY, never negative sentiment or adverse outcome.
"lost to", "sales fell", "was unsuccessful", "is illegal" are positive assertions.
When polarity is negative, make the relation affirmative: "does not have X" -> relation="has", object="X", polarity="negative".
When uncertain, preserve the modality and attribution; do not convert a possibility into a certainty.
Retain numbers, temporal scope, conditions, exceptions, comparison direction and every factual clause.
If a sentence is a fragmented title or punctuation, cite it together with the adjacent sentence supporting
the reconstructed fact. Keep the original source spelling; don't silently fix dates or names.
An empty object is allowed, an invented entity is not. Include all source sentence indices in the facts.
Do not copy a whole paragraph into a field. Correct all problematic rows, preserve valid rows.
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--model', default='qwen3.7-flash')
    parser.add_argument('--env-file', default='.env.local')
    args = parser.parse_args()
    read_env(Path(args.env_file))
    root = Path(args.root)
    documents = json.loads((root / 'text_documents.json').read_text(encoding='utf-8'))
    output = root / 'source_review_corrections.json'
    corrections = json.loads(output.read_text(encoding='utf-8')) if output.exists() else {}
    selected = []
    for doc in documents:
        if doc['id'] in corrections:
            validate_extraction(doc, corrections[doc['id']])
            continue
        fp = digest({'document': doc, 'prompt': PROMPT, 'model': args.model, 'endpoint': os.environ['LLM_BASE_URL'], 'enable_thinking': False})
        path = root / 'extraction_cache' / (fp + '.json')
        if not path.exists():
            path = root / 'extraction_cache' / (fp + '.attempt-2.json')
        result = json.loads(path.read_text(encoding='utf-8'))['result']
        error = ''
        try:
            validate_extraction(doc, result)
        except ValueError as exc:
            error = str(exc)
        if error or any(f.get('polarity') in ('negative', 'uncertain') for f in result['facts']):
            selected.append((doc, result, error))
    def repair(item):
        doc, previous, error = item
        client = AuditedChat(args.model, max_tokens=8192)
        for attempt in range(4):
            message = client.call('source_review', [{'role': 'system', 'content': REVIEW}, {'role': 'user', 'content':
                json.dumps({'source': doc, 'previous_extraction': previous, 'validation_error': error}, ensure_ascii=False)}])
            previous = json_object(message.get('content', ''))
            if isinstance(previous, dict) and 'facts' not in previous and isinstance(previous.get('previous_extraction'), dict):
                previous = previous['previous_extraction']
            (root / 'extraction_cache' / (doc['id'] + f'.review-{attempt}.json')).write_text(json.dumps(
                {'result': previous, 'call': client.calls[-1]}, ensure_ascii=False, indent=2), encoding='utf-8')
            try:
                validate_extraction(doc, previous)
                return doc, previous
            except ValueError as exc:
                error = str(exc)
        raise ValueError(doc['title'] + ': ' + error)
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(repair, item): item[0] for item in selected}
        for future in as_completed(futures):
            try:
                doc, result = future.result()
                corrections[doc['id']] = result
                output.write_text(json.dumps(corrections, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps({'reviewed': doc['title'], 'facts': len(result['facts'])}, ensure_ascii=True), flush=True)
            except Exception as exc:
                failures.append(str(exc))
                print(json.dumps({'failed': str(exc)}, ensure_ascii=True), flush=True)
    if failures:
        raise ValueError(failures)


if __name__ == '__main__':
    main()
