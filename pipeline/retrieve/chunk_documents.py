"""
Content-type-aware document chunker for the EDA corpus.

Three chunking strategies by content type:
  1. Code/HDL (.v, .sv, .tcl, .sdc): module/procedure boundaries, max 512 tokens
  2. Log/report (.log, .json reports): section boundaries, max 256 tokens
  3. Documentation (.md, .rst, .txt, forum JSONL): sliding window, max 384 tokens

Each chunk carries: source_file, chunk_index, content_type, token_count, source_id, text

Usage:
    python -m pipeline.retrieve.chunk_documents \
        --corpus C:\\eda-kg-data\\corpus\\staging\\dedup \
        --forums C:\\eda-kg-data\\corpus\\raw_docs\\forums \
        --orfs C:\\eda-kg-data\\orfs\\runs \
        --output data/chunks/chunks.jsonl
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from hashlib import md5


# Content type classification by extension
CODE_EXTENSIONS = {'.v', '.sv', '.vh', '.svh', '.tcl', '.sdc', '.lib', '.lef', '.def'}
LOG_EXTENSIONS = {'.log', '.rpt'}
DOC_EXTENSIONS = {'.md', '.rst', '.txt', '.html'}
DATA_EXTENSIONS = {'.json', '.csv', '.yaml', '.yml'}
SOURCE_EXTENSIONS = {'.py', '.cpp', '.cc', '.c', '.h', '.hpp', '.sh'}


def classify_content_type(filepath: str) -> str:
    """Classify file into content type for chunking strategy selection."""
    ext = Path(filepath).suffix.lower()
    name = Path(filepath).name.lower()

    if name == '6_report.json':
        return 'orfs_report'
    if ext in CODE_EXTENSIONS:
        return 'code'
    if ext in LOG_EXTENSIONS or name.endswith('_report.txt'):
        return 'log'
    if ext in DOC_EXTENSIONS:
        return 'documentation'
    if ext in SOURCE_EXTENSIONS:
        return 'source_code'
    if ext in DATA_EXTENSIONS:
        return 'data'
    return 'other'


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for code, ~3.5 for English."""
    return max(1, len(text) // 4)


def chunk_code(text: str, filepath: str, max_tokens: int = 512, overlap_tokens: int = 64) -> list:
    """Chunk code at module/procedure boundaries."""
    chunks = []
    # Split at module/procedure boundaries for Verilog/SystemVerilog
    ext = Path(filepath).suffix.lower()

    if ext in ('.v', '.sv', '.vh', '.svh'):
        # Split at module/endmodule boundaries
        pattern = r'(?=\b(?:module|package|interface|class)\b)'
    elif ext in ('.tcl', '.sdc'):
        # Split at proc definitions or major comments
        pattern = r'(?=\bproc\b|\n#{3,})'
    elif ext in ('.lef', '.def'):
        # Split at MACRO/PIN/LAYER blocks
        pattern = r'(?=\b(?:MACRO|PIN|LAYER|SITE|VIA)\b)'
    else:
        pattern = None

    if pattern:
        sections = re.split(pattern, text)
        sections = [s for s in sections if s.strip()]
    else:
        sections = [text]

    for section in sections:
        tokens = estimate_tokens(section)
        if tokens <= max_tokens:
            chunks.append(section)
        else:
            # Sub-chunk by lines with overlap
            lines = section.split('\n')
            current = []
            current_tokens = 0
            for line in lines:
                line_tokens = estimate_tokens(line)
                if current_tokens + line_tokens > max_tokens and current:
                    chunks.append('\n'.join(current))
                    # Keep overlap
                    overlap_lines = []
                    overlap_count = 0
                    for prev_line in reversed(current):
                        overlap_count += estimate_tokens(prev_line)
                        if overlap_count > overlap_tokens:
                            break
                        overlap_lines.insert(0, prev_line)
                    current = overlap_lines
                    current_tokens = overlap_count
                current.append(line)
                current_tokens += line_tokens
            if current:
                chunks.append('\n'.join(current))

    return chunks


def chunk_log(text: str, max_tokens: int = 256) -> list:
    """Chunk logs at section boundaries."""
    # Split at section markers
    section_pattern = r'(?=\[(?:INFO|WARNING|ERROR|WARN|DEBUG)\]|\n-{3,}|\n={3,}|\n\d{4}-\d{2}-\d{2})'
    sections = re.split(section_pattern, text)
    sections = [s for s in sections if s.strip()]

    chunks = []
    current = ""
    for section in sections:
        combined = current + section
        if estimate_tokens(combined) > max_tokens and current:
            chunks.append(current)
            current = section
        else:
            current = combined
    if current.strip():
        chunks.append(current)

    # Sub-chunk any oversized sections
    final = []
    for chunk in chunks:
        if estimate_tokens(chunk) > max_tokens * 1.5:
            lines = chunk.split('\n')
            sub = []
            sub_tokens = 0
            for line in lines:
                lt = estimate_tokens(line)
                if sub_tokens + lt > max_tokens and sub:
                    final.append('\n'.join(sub))
                    sub = []
                    sub_tokens = 0
                sub.append(line)
                sub_tokens += lt
            if sub:
                final.append('\n'.join(sub))
        else:
            final.append(chunk)
    return final


def chunk_documentation(text: str, max_tokens: int = 384, overlap_tokens: int = 96) -> list:
    """Sentence-aware sliding window for documentation."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    if len(sentences) <= 1:
        sentences = text.split('\n')

    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = estimate_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(' '.join(current) if not any('\n' in s for s in current)
                         else '\n'.join(current))
            # Overlap
            overlap = []
            ot = 0
            for prev in reversed(current):
                ot += estimate_tokens(prev)
                if ot > overlap_tokens:
                    break
                overlap.insert(0, prev)
            current = overlap
            current_tokens = ot
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(' '.join(current) if not any('\n' in s for s in current)
                     else '\n'.join(current))
    return chunks


def chunk_forum_qa(records: list) -> list:
    """Chunk forum Q&A pairs — keep question + answer together."""
    chunks = []
    for record in records:
        # Support both GitHub issue format and our custom format
        title = record.get('question_title', record.get('title', ''))
        body = record.get('question_body', record.get('body', ''))
        answer = record.get('answer', '')
        answers = record.get('answers', [])

        text_parts = []
        if title:
            text_parts.append(f"Q: {title}")
        if body:
            text_parts.append(body[:1500])
        if answer:
            text_parts.append(f"A: {answer[:2000]}")
        for ans in answers[:3]:
            ans_body = ans.get('body', '')[:1000]
            if ans_body:
                text_parts.append(f"A: {ans_body}")

        text = '\n\n'.join(text_parts)
        tokens = estimate_tokens(text)

        if tokens <= 512:
            chunks.append({
                'text': text,
                'token_count': tokens,
                'content_type': 'forum_qa',
            })
        else:
            # Split at answer boundaries
            sub_chunks = chunk_documentation(text, max_tokens=384, overlap_tokens=96)
            for sc in sub_chunks:
                chunks.append({
                    'text': sc,
                    'token_count': estimate_tokens(sc),
                    'content_type': 'forum_qa',
                })
    return chunks


def chunk_orfs_report(text: str, filepath: str) -> list:
    """Chunk ORFS 6_report.json as structured data."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [{'text': text[:2048], 'token_count': estimate_tokens(text[:2048]),
                 'content_type': 'orfs_report'}]

    # Group metrics by category
    categories = {}
    for key, value in data.items():
        parts = key.split('__')
        cat = parts[0] if parts else 'other'
        if cat not in categories:
            categories[cat] = {}
        categories[cat][key] = value

    chunks = []
    for cat, metrics in categories.items():
        text = json.dumps({cat: metrics}, indent=2)
        chunks.append({
            'text': text,
            'token_count': estimate_tokens(text),
            'content_type': 'orfs_report',
        })
    return chunks


def process_corpus(corpus_dir: str) -> list:
    """Process all corpus files."""
    all_chunks = []
    corpus_path = Path(corpus_dir)
    files = list(corpus_path.rglob('*'))
    files = [f for f in files if f.is_file() and f.stat().st_size > 0]

    print(f"Processing {len(files)} corpus files...")
    skipped = 0

    for i, filepath in enumerate(files):
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{len(files)} files processed, {len(all_chunks)} chunks so far")

        try:
            text = filepath.read_text(encoding='utf-8', errors='replace')
        except Exception:
            skipped += 1
            continue

        if len(text.strip()) < 20:
            skipped += 1
            continue

        content_type = classify_content_type(str(filepath))
        source_id = md5(str(filepath).encode()).hexdigest()[:12]

        if content_type == 'code':
            raw_chunks = chunk_code(text, str(filepath))
        elif content_type == 'source_code':
            raw_chunks = chunk_code(text, str(filepath), max_tokens=512)
        elif content_type == 'log':
            raw_chunks = chunk_log(text)
        elif content_type == 'orfs_report':
            orfs_chunks = chunk_orfs_report(text, str(filepath))
            for j, c in enumerate(orfs_chunks):
                c['source_file'] = str(filepath)
                c['chunk_index'] = j
                c['source_id'] = source_id
            all_chunks.extend(orfs_chunks)
            continue
        elif content_type in ('documentation', 'other'):
            raw_chunks = chunk_documentation(text)
        elif content_type == 'data':
            raw_chunks = chunk_log(text, max_tokens=256)
        else:
            raw_chunks = chunk_documentation(text)

        for j, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) < 10:
                continue
            all_chunks.append({
                'source_file': str(filepath),
                'chunk_index': j,
                'content_type': content_type,
                'token_count': estimate_tokens(chunk_text),
                'source_id': source_id,
                'text': chunk_text,
            })

    print(f"  Skipped: {skipped} files (empty or unreadable)")
    return all_chunks


def process_forums(forums_dir: str) -> list:
    """Process forum JSONL files."""
    all_chunks = []
    forums_path = Path(forums_dir)

    for jsonl_file in forums_path.glob('*.jsonl'):
        print(f"  Processing {jsonl_file.name}...")
        records = []
        with open(jsonl_file, encoding='utf-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        qa_chunks = chunk_forum_qa(records)
        for j, chunk in enumerate(qa_chunks):
            chunk['source_file'] = str(jsonl_file)
            chunk['chunk_index'] = j
            chunk['source_id'] = md5(str(jsonl_file).encode()).hexdigest()[:12]
        all_chunks.extend(qa_chunks)

    return all_chunks


def process_orfs(orfs_dir: str) -> list:
    """Process ORFS report files."""
    all_chunks = []
    orfs_path = Path(orfs_dir)

    for report_file in orfs_path.rglob('*'):
        if not report_file.is_file():
            continue
        try:
            text = report_file.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue

        if len(text.strip()) < 20:
            continue

        content_type = classify_content_type(str(report_file))
        source_id = md5(str(report_file).encode()).hexdigest()[:12]

        if content_type == 'orfs_report':
            orfs_chunks = chunk_orfs_report(text, str(report_file))
            for j, c in enumerate(orfs_chunks):
                c['source_file'] = str(report_file)
                c['chunk_index'] = j
                c['source_id'] = source_id
            all_chunks.extend(orfs_chunks)
        else:
            if content_type == 'log':
                raw_chunks = chunk_log(text)
            else:
                raw_chunks = chunk_documentation(text, max_tokens=256)

            for j, chunk_text in enumerate(raw_chunks):
                if len(chunk_text.strip()) < 10:
                    continue
                all_chunks.append({
                    'source_file': str(report_file),
                    'chunk_index': j,
                    'content_type': content_type,
                    'token_count': estimate_tokens(chunk_text),
                    'source_id': source_id,
                    'text': chunk_text,
                })

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk EDA corpus for vector embedding")
    parser.add_argument('--corpus', default=r'C:\eda-kg-data\corpus\staging\dedup')
    parser.add_argument('--forums', default=r'C:\eda-kg-data\corpus\raw_docs\forums')
    parser.add_argument('--orfs', default=r'C:\eda-kg-data\orfs\runs')
    parser.add_argument('--output', default='data/chunks/chunks.jsonl')
    args = parser.parse_args()

    all_chunks = []

    # 1. Corpus files
    if os.path.isdir(args.corpus):
        print(f"=== Corpus: {args.corpus} ===")
        all_chunks.extend(process_corpus(args.corpus))
        print(f"  Corpus chunks: {len(all_chunks)}")

    # 2. Forum Q&A
    if os.path.isdir(args.forums):
        print(f"\n=== Forums: {args.forums} ===")
        forum_chunks = process_forums(args.forums)
        all_chunks.extend(forum_chunks)
        print(f"  Forum chunks: {len(forum_chunks)}")

    # 3. ORFS reports
    if os.path.isdir(args.orfs):
        print(f"\n=== ORFS reports: {args.orfs} ===")
        orfs_chunks = process_orfs(args.orfs)
        all_chunks.extend(orfs_chunks)
        print(f"  ORFS chunks: {len(orfs_chunks)}")

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    # Summary
    from collections import Counter
    type_counts = Counter(c['content_type'] for c in all_chunks)
    total_tokens = sum(c['token_count'] for c in all_chunks)
    oversized = sum(1 for c in all_chunks if c['token_count'] > 512)

    print(f"\n{'='*60}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Oversized (>512 tokens): {oversized}")
    print(f"\nBy content type:")
    for ct, count in type_counts.most_common():
        print(f"  {ct}: {count}")
    print(f"\nEstimated embedding cost (voyage-code-2 @ $0.12/1M tokens): ${total_tokens / 1_000_000 * 0.12:.2f}")
    print(f"Output: {out_path}")

    # Validate
    required = ['source_file', 'chunk_index', 'content_type', 'token_count', 'source_id']
    missing = sum(1 for c in all_chunks if not all(k in c for k in required))
    assert oversized < len(all_chunks) * 0.15, f"Too many oversized chunks: {oversized}/{len(all_chunks)}"
    assert missing == 0, f"Chunks missing required fields: {missing}"
    print("Chunking gate: PASS")


if __name__ == '__main__':
    main()
