"""Audit and publish the complete #176 matrix after model-free CLI rescore.

Run from the repository root:
    python tools/report_full_matrix.py results/full-matrix-20261021

Native per-question files are read only. This script refuses incomplete,
mislabelled or changed inputs rather than publishing partial matrix tables.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.artifacts import dump_json
from src.evaluation.full_ablation import (
    MATRIX, METRICS, answer_summary, check_transactions, read_rows, scored_for,
    summarize, validate_snapshot, ranking_sources,
)
from src.evaluation.metrics import score_question
from src.evaluation.records import BenchmarkQuestion, RunResult
from src.rag.constants import MAX_OUTPUT_TOKENS


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + '/', '').replace(str(Path.home()) + '/', '$HOME/')
    return value


def verify_committed_source(provenance: dict, *, current: bool) -> None:
    for name, expected in provenance['source_files_sha256'].items():
        committed = subprocess.check_output(['git', 'show', f"{provenance['source_commit']}:{name}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != expected or (current and digest(ROOT / name) != expected):
            raise ValueError(f'measurement source changed: {name}')


def verify_replay_sources(retrieval: dict, answers: dict) -> None:
    """Permit answer-only credential validation changes, never changed rankings."""
    if ranking_sources(retrieval) != ranking_sources(answers):
        raise ValueError('ranking source differs between retrieval and answer replay')
    path = 'src/evaluation/full_ablation.py'
    old_dependencies = {name: value for name, value in retrieval['source_files_sha256'].items() if name != path}
    new_dependencies = {name: value for name, value in answers['source_files_sha256'].items() if name != path}
    if old_dependencies != new_dependencies:
        raise ValueError('a non-runner dependency changed between retrieval and answer replay')
    def definitions(commit):
        content = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)
        nodes = ast.parse(content).body
        result = {}
        for node in nodes:
            name = getattr(node, 'name', None)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MATRIX' for t in node.targets):
                name = 'MATRIX'
            if name:
                result[name] = ast.dump(node, include_attributes=False)
        return result
    old, new = (definitions(p['source_commit']) for p in (retrieval, answers))
    for name in ('MatrixConfig', 'MATRIX', 'scored_for', 'query_record', 'ranking',
                 'CachedPart', 'measure_question', 'ReplayRetriever', 'verified_baseline'):
        if old[name] != new[name]:
            raise ValueError(f'retrieval/replay definition changed across phase sources: {name}')


def usage_summary(costs: list[dict], configuration: dict) -> dict:
    """Retain unknown failed-request usage; never present partial tokens as total."""
    result = {key: configuration[key] for key in ('provider', 'model')}
    result.update({key: sum(c[key] for c in costs)
                   for key in ('model_calls', 'successful_calls', 'usage_calls')})
    result['unsuccessful_attempts'] = result['model_calls'] - result['successful_calls']
    for key in ('input_tokens', 'output_tokens', 'observed_input_tokens', 'observed_output_tokens', 'usd'):
        known = [c[key] for c in costs if c[key] is not None]
        result[key] = {'known_total': sum(known) if known else None,
                       'known_queries': len(known), 'unknown_queries': len(costs) - len(known)}
    return result


def audit(run_dir: Path) -> tuple[dict, dict]:
    config = read_json(run_dir / 'config.json')
    provenance = config['provenance']
    declaration = config['benchmark_declaration']
    if not declaration['required'] or not declaration['matches_decision_benchmark']:
        raise ValueError('a required and matching decision benchmark declaration is necessary')
    if tuple(config['matrix']) != tuple(MATRIX) or config['matrix'] != {cid: row.to_dict() for cid, row in MATRIX.items()}:
        raise ValueError('all seven current C0-C6 configurations must be frozen')
    questions = [BenchmarkQuestion.from_mapping(q) for q in read_rows(run_dir / 'benchmark.jsonl')]
    validate_snapshot(config, questions)
    if len(questions) != 12729 or Counter(q.benchmark for q in questions) != {'team': 150, 'xbrl': 12579}:
        raise ValueError('complete 150 + 12,579 benchmark required')
    accepted = read_json(ROOT / 'docs/evaluation/c5-full-20261017-manifest.json')['experiment']['provenance']
    for key in ('questions', 'by_benchmark', 'question_records_sha256', 'retrieval_settings'):
        if provenance[key] != accepted[key]:
            raise ValueError(f'accepted full benchmark/settings differ: {key}')
    if provenance['corpus']['records_sha256'] != accepted['corpus']['records_sha256']:
        raise ValueError('corpus differs from the accepted full C5 study')
    if [p['sha256'] for p in provenance['inputs']] != [p['sha256'] for p in accepted['inputs']]:
        raise ValueError('input file hashes differ from the accepted full study')
    source = provenance['source_commit']
    verify_committed_source(provenance, current=False)
    transactions = read_rows(run_dir / 'pairs.jsonl')
    check_transactions(transactions, questions, tuple(MATRIX))
    if len(transactions) != len(questions):
        raise ValueError('retrieval is incomplete')
    for question, transaction in zip(questions, transactions):
        for cid, row in transaction['rows'].items():
            if row['config'] != config['matrix'][cid] | {'candidate_k': 50}:
                raise ValueError('per-question configuration differs')
            result = RunResult(question.question_id, row['retriever'], tuple(row['retrieved_chunk_ids']),
                               tuple(row['retrieved_scores']), row['latency_ms'])
            metrics = score_question(scored_for(question, row), result, k=16)
            if any(row[key] != metrics[key] for key in METRICS):
                raise ValueError('per-question retrieval metrics differ from saved rankings')
        c4, c6 = transaction['rows']['C4'], transaction['rows']['C6']
        if c6.get('rerank_diagnostics') is not None or any(s.get('rerank_diagnostics') is not None for s in c6['sub_results']):
            raise ValueError('C6 unexpectedly reranked evidence')
        if not c6['sub_results'] and any(c6[key] != c4[key] for key in ('retrieved_chunk_ids', 'retrieved_scores', 'latency_ms', *METRICS)):
            raise ValueError('unsplit C6 differs from C4')
    retrieval = read_json(run_dir / 'summary.json')
    rescored = read_json(run_dir / 'rescored/summary.json')
    if not retrieval['complete'] or retrieval['configurations'] != rescored['configurations']:
        raise ValueError('retrieval aggregates do not match model-free rescore')
    team = [q.question_id for q in questions if q.benchmark == 'team']
    answer_report = read_json(run_dir / 'answer-summary.json')
    if not answer_report['complete'] or answer_report != read_json(run_dir / 'rescored/answer-summary.json'):
        raise ValueError('complete answer aggregates do not match model-free rescore')
    answer_config = read_json(run_dir / 'answers-config.json')
    verify_committed_source(answer_config['source'], current=True)
    verify_replay_sources(provenance, answer_config['source'])
    for phase, provider, model in (('generation', 'soclaas', 'qwen3.6:35b'),
                                   ('judge_generation', 'mistral', 'ministral-8b-2512')):
        if (answer_config[phase]['provider'] != provider or answer_config[phase]['model'] != model
                or answer_config[phase]['temperature'] != 0):
            raise ValueError(f'answer phase uses a different provider/model/temperature: {phase}')
    if answer_config['judge_rubric'] != 'answer-quality-v1' or answer_config['judge_generation']['prompt_template_id'] != 'answer-quality-v1':
        raise ValueError('actual judge rubric differs from the declared scoring setup')
    if answer_config['retrieval_config_sha256'] != digest(run_dir / 'config.json'):
        raise ValueError('answers refer to a different retrieval run')
    if answer_config['questions'] != [q.to_dict() for q in questions if q.benchmark == 'team']:
        raise ValueError('answer cohort differs from frozen team questions')
    methods, verification, verification_status, denominators, usage = {}, {}, {}, {}, {}
    breakdown = {benchmark: {} for benchmark in ('team', 'xbrl')}
    for cid in MATRIX:
        rows = [t['rows'][cid] for t in transactions]
        if read_rows(run_dir / cid / 'questions.jsonl') != rows:
            raise ValueError('exported per-question retrieval differs from transactions')
        summary = next(row for row in retrieval['configurations'] if row['config']['id'] == cid)
        if any(summary[key] != value for key, value in summarize(rows).items()):
            raise ValueError('native retrieval summary differs from transactions')
        for benchmark in breakdown:
            selected = [r for q, r in zip(questions, rows) if q.benchmark == benchmark]
            breakdown[benchmark][cid] = summarize(selected)
        answers = read_rows(run_dir / cid / 'answers.jsonl')
        if [r['question_id'] for r in answers] != team:
            raise ValueError(f'incomplete or reordered {cid} answers')
        if any(row['matrix_config'] != config['matrix'][cid] | {'candidate_k': 50}
               or row['answer']['config'] != answer_config['generation'] for row in answers):
            raise ValueError('answer configuration differs from the measured setup')
        expected_summary = answer_summary(answers, cid)
        if read_json(run_dir / cid / 'answer-summary.json') != expected_summary or answer_report['configurations'][cid] != expected_summary:
            raise ValueError('answer aggregate differs from per-question outputs')
        methods[cid] = dict(Counter(r['quality']['method'] for r in answers))
        denominators[cid] = {'total': len(answers), 'llm_judged': methods[cid].get('llm_judge', 0),
                             'automatically_scored': methods[cid].get('abstention', 0),
                             'judge_failed': methods[cid].get('judge_failed', 0),
                             'not_scored': methods[cid].get('not_scored', 0)}
        if sum(denominators[cid][key] for key in ('llm_judged', 'automatically_scored', 'judge_failed', 'not_scored')) != len(answers):
            raise ValueError('unrecognized scoring method')
        usage[cid] = {
            'generation': usage_summary([r['cost'] for r in answers], answer_config['generation']),
            'judge': usage_summary([r['judge_overhead']['cost'] for r in answers], answer_config['judge_generation']),
        }
        verification[cid] = sum(r['answer']['verification'] is not None for r in answers)
        if verification[cid] != (150 if cid == 'C6' else 0):
            raise ValueError('verification was not confined to C6')
        verification_status[cid] = dict(Counter(check['status'] for row in answers
            if row['answer']['verification'] for check in row['answer']['verification']['checks']))
    runtime = read_json(run_dir / 'runtime.json')
    native = {p.relative_to(run_dir).as_posix(): digest(p) for p in sorted(run_dir.rglob('*'))
              if p.is_file() and p.suffix != '.lock' and not p.name.endswith('process-status.json')
              and 'rescored' not in p.relative_to(run_dir).parts}
    archived = {}
    for pattern in ('full-matrix-20261017-*.json', 'c5-full-20261017-*.json', 'c6-rebase-full-20261019-*.json'):
        for path in (ROOT / 'docs/evaluation').glob(pattern):
            relative = path.relative_to(ROOT).as_posix()
            committed = subprocess.check_output(['git', 'show', f'{source}:{relative}'], cwd=ROOT)
            if hashlib.sha256(committed).hexdigest() != digest(path):
                raise ValueError(f'archived result changed: {relative}')
            archived[relative] = digest(path)
    report = {
        'run_id': run_dir.name, 'configurations_measured': list(MATRIX),
        'retrieval_questions_per_row': len(questions), 'by_benchmark': provenance['by_benchmark'],
        'team_answers_per_row': len(team), 'reference_imported': config['reference'] is not None,
        'benchmark_declaration': declaration, 'retrieval': retrieval, 'retrieval_by_benchmark': breakdown,
        'answers': answer_report, 'evaluation_methods': methods, 'verification_outputs': verification,
        'scoring_denominators': denominators, 'usage_by_provider': usage,
        'generation': answer_config['generation'], 'judge': answer_config['judge_generation'],
        'judge_rubric': answer_config['judge_rubric'],
        'max_output_tokens_per_call': MAX_OUTPUT_TOKENS,
        'verification_claim_status_counts': verification_status,
        'decomposition_questions': sum(bool(t['rows']['C6']['sub_results']) for t in transactions),
        'limitations': [
            'XBRL retrieval uses facts-derived relevance proxies, not independent answer labels; answers are evaluated on team questions only.',
            'C0 uses reconstructed fixed-window evidence with projected gold labels in a different chunk namespace.',
            'SOCLAAS generation and Mistral judging are separate hosted calls; semantic scores are model judgements, not human accuracy.',
            'Hosted models are nondeterministic; separate generations on identical evidence do not isolate verification benefits.',
            'C6 verification annotates claims and does not rewrite answers.',
            'Laptop retrieval overlaps hosted answer evaluation and uncontrolled local applications; timings are not isolated throughput.',
            'The archived Mistral-generator studies use different generator settings and are not a controlled comparison to this SOCLAAS study.',
        ],
    }
    manifest = {
        'run_id': run_dir.name, 'experiment': portable(config), 'answers_config': portable(answer_config),
        'answer_settings': {
            'generation': dict(answer_config['generation'], max_output_tokens=MAX_OUTPUT_TOKENS),
            'judge': dict(answer_config['judge_generation'], max_output_tokens=MAX_OUTPUT_TOKENS),
        },
        'runtime_segments': [portable(read_json(p)) for p in sorted(run_dir.glob('runtime-*.json'))],
        'answer_runtime_segments': [portable(read_json(p)) for p in sorted(run_dir.glob('answers-runtime-*.json'))],
        'native_artifacts_sha256': native, 'archived_artifacts_unchanged_sha256': archived,
        'report_script': {'path': 'tools/report_full_matrix.py', 'sha256': digest(Path(__file__))},
        'validation': {'complete_declared_benchmark': True, 'committed_source_verified': True,
                       'per_question_metrics_recomputed': True, 'exact_retrieval_rescore': True,
                       'exact_answer_rescore': True, 'team_answer_order_verified': True},
        'reproduce_at_source': {'retrieval': source, 'answers': answer_config['source']['source_commit']},
        'report_script_note': 'Use the report helper from this PR reporting commit while keeping src/ at the frozen answer hashes; the measurement commits predate the reporting helper.',
        'shareable_note': 'Repository/home paths normalized. Native per-question artifacts remain in the ignored run directory with hashes recorded here.',
    }
    proof = run_dir / 'review-validation.json'
    if proof.exists():
        validation = read_json(proof)
        for name, expected in validation['preserved_native_artifacts_sha256'].items():
            if digest(run_dir / name) != expected:
                raise ValueError(f'completed resume changed {name}')
        manifest['validation'].update(validation)
    return portable(report), manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs/evaluation')
    args = parser.parse_args()
    report, manifest = audit(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / f'{args.run_dir.name}-summary.json', report)
    dump_json(args.output_dir / f'{args.run_dir.name}-manifest.json', manifest)
    print(json.dumps({key: report[key] for key in ('run_id', 'retrieval_questions_per_row', 'team_answers_per_row')}))


if __name__ == '__main__':
    main()
