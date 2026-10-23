"""Scientific reporting guards for the independently frozen matrix phases."""
import pytest

from src.evaluation.quality import query_cost
from tools import report_full_matrix as report


def test_usage_report_keeps_partial_failed_request_tokens_separate_from_complete_totals():
    failed = query_cost('mistral', [{'input_tokens': 7, 'output_tokens': 3}],
                        attempts=2, successful_calls=1, prices=None)
    unused = query_cost('mistral', [], attempts=0, prices=None)
    result = report.usage_summary([failed, unused], {'provider': 'mistral', 'model': 'judge'})
    assert result['model_calls'] == 2 and result['unsuccessful_attempts'] == 1
    assert result['input_tokens'] == {'known_total': 0, 'known_queries': 1, 'unknown_queries': 1}
    assert result['observed_input_tokens'] == {'known_total': 7, 'known_queries': 2, 'unknown_queries': 0}
    assert result['usd'] == {'known_total': 0.0, 'known_queries': 1, 'unknown_queries': 1}


@pytest.mark.parametrize('change', ['answer_validation_only', 'ranking_function', 'ranking_dependency', 'decomposition_dependency'])
def test_phase_audit_allows_answer_validation_changes_but_refuses_changed_rankings(monkeypatch, change):
    names = ('scored_for', 'query_record', 'ranking', 'CachedPart', 'measure_question',
             'ReplayRetriever', 'verified_baseline')
    code = 'class MatrixConfig: pass\nMATRIX = {}\n' + ''.join(f'def {name}():\n    return 1\n' for name in names)
    newer = code + 'def answers():\n    validate_both_keys()\n'
    if change == 'ranking_function':
        newer = newer.replace('def measure_question():\n    return 1', 'def measure_question():\n    return 2')
    sources = {'src/retrieval/hybrid.py': 'same', 'src/retrieval/rerank.py': 'same',
               'src/evaluation/run.py': 'same', 'src/rag/query.py': 'same', 'src/stack.py': 'same',
               'src/rag/decompose.py': 'same'}
    old = {'source_commit': 'old', 'source_files_sha256': dict(sources)}
    new = {'source_commit': 'new', 'source_files_sha256': dict(sources)}
    if change == 'ranking_dependency':
        new['source_files_sha256']['src/retrieval/hybrid.py'] = 'changed'
    if change == 'decomposition_dependency':
        new['source_files_sha256']['src/rag/decompose.py'] = 'changed'
    def committed(args, **kwargs):
        assert args[:2] == ['git', 'show']
        return (code if args[-1].startswith('old:') else newer).encode()
    monkeypatch.setattr(report.subprocess, 'check_output', committed)
    if change == 'answer_validation_only':
        report.verify_replay_sources(old, new)
    else:
        with pytest.raises(ValueError, match='ranking source|retrieval/replay definition|non-runner dependency'):
            report.verify_replay_sources(old, new)
