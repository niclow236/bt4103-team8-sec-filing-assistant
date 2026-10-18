"""Shared JSONL recovery and atomic report replacement."""
import gzip
import io
import json

import pytest

from src.evaluation.artifacts import dump_json, read_jsonl
from src.evaluation.full_ablation import read_rows
from src.evaluation.rerank_ablation import read_pairs


@pytest.mark.parametrize('compressed',[False,True])
@pytest.mark.parametrize('tail',['{"partial":','{"valid_but_unfinished":true}'])
def test_plain_and_gzip_readers_recover_only_complete_lines(tmp_path,compressed,tail):
    path=tmp_path/'rows'
    text='{"complete":true}\n'+tail
    if compressed:
        with gzip.open(path,'wt') as stream:stream.write(text)
        read=read_pairs
    else:
        path.write_text(text)
        read=read_rows
    with pytest.raises(EOFError):read(path)
    assert read(path,recover=True)==[{'complete':True}]


def test_recovery_does_not_hide_a_complete_invalid_line():
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(io.StringIO('{"bad":\n'),recover=True)


def test_atomic_json_write_preserves_previous_artifact_when_replace_fails(tmp_path,monkeypatch):
    path=tmp_path/'summary.json'
    path.write_text('original')
    def fail(*args):raise OSError('replace failed')
    monkeypatch.setattr(type(path),'replace',fail)
    with pytest.raises(OSError,match='replace failed'):dump_json(path,{'new':True})
    assert path.read_text()=='original'
