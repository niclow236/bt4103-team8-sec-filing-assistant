"""Exercise the actual matrix commands across interruption/replay boundaries."""
import importlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from filelock import FileLock, Timeout

from src.evaluation import full_ablation as matrix
from src.evaluation.harness import RunInterrupted, RunStopped
from src.evaluation.provenance import record_provenance
from src.evaluation.quality import query_cost
from src.rag.constants import PROMPT_TEMPLATE_ID
from src.rag.records import Answer, GenerationConfig
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import CrossEncoderReranker
from conftest import RerankModel
from test_full_ablation import Part, chunk, question


def artifacts(directory):
    return {p.relative_to(directory).as_posix():p.read_bytes()
            for p in directory.rglob('*') if p.is_file() and p.suffix!='.lock'}


@pytest.fixture
def runner(tmp_path,monkeypatch):
    snapshot=[dict(chunk('a','AAPL'),chunk_budget=1800,chunk_overlap=300),
              dict(chunk('b','MSFT'),chunk_budget=1800,chunk_overlap=300)]
    questions=[question('q1'),replace(question('q2'),question='What was Apple revenue in FY2024?',
                                    ticker='AAPL',supporting_chunk_ids=('a',),question_type='numeric')]
    inputs=tmp_path/'questions.jsonl'
    matrix.write_rows(inputs,[q.to_dict() for q in questions])
    state=SimpleNamespace(snapshot=snapshot,questions=questions,reads=0,seen=[],fail_after=None,
                          calls=[],build='original',revision='dense-r1',platform=None,empty=False)
    def no_read(**kwargs):
        pytest.fail('a second independent corpus read escaped the frozen snapshot')
    monkeypatch.setattr('src.evaluation.provenance.iter_chunks',no_read)
    monkeypatch.setattr('src.evaluation.run.iter_chunks',no_read)
    monkeypatch.setattr('src.evaluation.benchmark.iter_chunks',no_read)
    # Retrieval materializes its own list; this checks identity across consumers, not the fixture's input list.
    def snapshot_read(**kwargs):
        state.reads+=1
        return iter(snapshot)
    monkeypatch.setattr(matrix,'iter_chunks',snapshot_read)
    def provenance(inputs,questions,processed_dir,*,chunks=None):
        assert chunks==snapshot
        state.seen.append(chunks)
        value=record_provenance(inputs,questions,processed_dir,chunks=chunks)
        if state.platform is not None: value['environment']['platform']=state.platform
        return value
    monkeypatch.setattr(matrix,'record_provenance',provenance)
    original_fixed=matrix.FixedSizeBM25Retriever
    def fixed(processed_dir,*,chunks):
        assert chunks is state.seen[-1]
        return original_fixed(processed_dir,chunks=chunks)
    monkeypatch.setattr(matrix,'FixedSizeBM25Retriever',fixed)
    def build(*args,**kwargs):
        sparse,dense=Part(snapshot),Part(snapshot)
        dense.name='dense'
        dense.runtime_info={'device':'cpu','revision':state.revision}
        hybrid=HybridRetriever(sparse,dense)
        if state.empty: hybrid.search=lambda *a,**kw:[]
        return hybrid
    monkeypatch.setattr(matrix,'build_retriever',build)
    monkeypatch.setattr(matrix,'_index_manifests',lambda hybrid:{
        'bm25_manifest':{'built_at':state.build},'dense_manifest':{'built_at':state.build}})
    def reranker(config):
        model=RerankModel([])
        model.predict=lambda pairs,**kwargs:[float(i) for i in range(len(pairs))]
        return CrossEncoderReranker(config,model=model)
    monkeypatch.setattr(matrix,'CrossEncoderReranker',reranker)
    original_measure=matrix.measure_question
    def measure(q,*args,**kwargs):
        if state.fail_after is not None and len(state.calls)==state.fail_after:
            raise OSError('interrupted measurement')
        state.calls.append(q.question_id)
        return original_measure(q,*args,**kwargs)
    monkeypatch.setattr(matrix,'measure_question',measure)
    args=SimpleNamespace(questions=[inputs],processed_dir=tmp_path,device='cpu',reference=None,
                         reference_pairs_sha256=None,results_root=tmp_path/'results',run_id='matrix',
                         workload_note='first segment',resume=False)
    state.args=args
    state.directory=args.results_root/args.run_id
    state.real_fixed=original_fixed
    return state


@pytest.mark.parametrize('difference',['counts','content','order'])
def test_required_decision_benchmark_is_rejected_before_index_construction(runner,monkeypatch,difference):
    questions=runner.questions
    monkeypatch.setitem(matrix.POLICY,'decision_benchmark_counts',{'team':len(questions)})
    monkeypatch.setitem(matrix.POLICY,'decision_question_records_sha256',matrix.question_records_sha256(questions))
    if difference=='counts':
        supplied=questions[:1]
    elif difference=='content':
        supplied=[replace(questions[0],expected_answer='Changed answer'),questions[1]]
    else:
        supplied=list(reversed(questions))
    matrix.write_rows(runner.args.questions[0],[q.to_dict() for q in supplied])
    runner.args.require_decision_benchmark=True
    monkeypatch.setattr(matrix,'build_retriever',lambda *a,**kw:pytest.fail('constructed an index for a wrong benchmark'))
    with pytest.raises(ValueError,match='exact frozen question counts and digest'):
        matrix.retrieval(runner.args)
    assert not runner.directory.exists() and not runner.calls


def test_required_decision_benchmark_is_frozen_and_checked_on_replay(runner,monkeypatch):
    monkeypatch.setitem(matrix.POLICY,'decision_benchmark_counts',{'team':len(runner.questions)})
    monkeypatch.setitem(matrix.POLICY,'decision_question_records_sha256',matrix.question_records_sha256(runner.questions))
    runner.args.require_decision_benchmark=True
    matrix.retrieval(runner.args)
    config=json.loads((runner.directory/'config.json').read_text())
    declaration=config['benchmark_declaration']
    assert declaration['required'] and declaration['matches_decision_benchmark']
    assert declaration['counts']=={'team':2}
    matrix.validate_snapshot(config,runner.questions)
    # Replay uses its frozen policy even if a later release declares a new cohort.
    monkeypatch.setitem(matrix.POLICY,'decision_question_records_sha256','0'*64)
    matrix.validate_snapshot(config,runner.questions)
    for key,value in [('counts',{'team':1}),('matches_decision_benchmark',False),
                      ('decision_question_records_sha256','0'*64)]:
        changed=deepcopy(config)
        changed['benchmark_declaration'][key]=value
        with pytest.raises(ValueError,match='benchmark declaration'):
            matrix.validate_snapshot(changed,runner.questions)


def test_diagnostic_subset_is_explicitly_recorded_without_claiming_the_decision_benchmark(runner):
    matrix.retrieval(runner.args)
    declaration=json.loads((runner.directory/'config.json').read_text())['benchmark_declaration']
    assert declaration['counts']=={'team':2}
    assert not declaration['required'] and not declaration['matches_decision_benchmark']


@pytest.mark.parametrize('count',[0,1])
def test_retrieval_runs_and_resumes_the_actual_transaction_loop(runner,count):
    runner.fail_after=count
    with pytest.raises(OSError,match='interrupted measurement'):
        matrix.retrieval(runner.args)
    original_runtime=(runner.directory/'runtime.json').read_bytes()
    original_pairs=matrix.read_rows(runner.directory/'pairs.jsonl')
    assert len(original_pairs)==count
    runner.fail_after=None
    runner.args.resume=True
    runner.args.workload_note='resumed overnight'
    original_path=runner.args.questions[0]
    runner.args.questions=[Path(os.path.relpath(original_path,Path.cwd()))]
    assert str(runner.args.questions[0])!=str(original_path)
    assert runner.args.questions[0].resolve()==original_path.resolve()
    runner.platform='new OS build string'
    matrix.retrieval(runner.args)
    assert runner.reads==2
    pairs=matrix.read_rows(runner.directory/'pairs.jsonl')
    assert pairs[:count]==original_pairs
    assert [r['question']['question_id'] for r in pairs]==['q1','q2']
    assert (runner.directory/'runtime.json').read_bytes()==original_runtime
    segment=json.loads((runner.directory/'runtime-0001.json').read_text())
    assert segment['resumed_after_questions']==count and segment['workload_note']=='resumed overnight'
    assert segment['environment']['platform']=='new OS build string'
    assert segment['inputs'][0]['path']==str(runner.args.questions[0])
    assert json.loads((runner.directory/'summary.json').read_text())['complete']
    before=artifacts(runner.directory)
    matrix.retrieval(runner.args)
    assert artifacts(runner.directory)==before


@pytest.mark.parametrize('change',['index','revision','missing_runtime'])
@pytest.mark.parametrize('count',[0,1])
def test_partial_resume_guards_runtime_identity_before_writing(runner,change,count):
    runner.fail_after=count
    with pytest.raises(OSError): matrix.retrieval(runner.args)
    runner.fail_after=None
    runner.args.resume=True
    if change=='index':runner.build='rebuilt'
    elif change=='revision':runner.revision='different'
    else:(runner.directory/'runtime.json').unlink()
    before=artifacts(runner.directory)
    if change=='missing_runtime' and count==0:
        matrix.retrieval(runner.args)
        assert json.loads((runner.directory/'runtime.json').read_text())['resumed_after_questions']==0
    else:
        with pytest.raises(ValueError,match='runtime model or index identity|original runtime'):
            matrix.retrieval(runner.args)
        assert artifacts(runner.directory)==before


@pytest.mark.parametrize('complete',[False,True])
def test_c0_mutation_is_refused_before_partial_or_completed_resume(runner,complete):
    if not complete:runner.fail_after=1
    try:matrix.retrieval(runner.args)
    except OSError:pass
    baseline=matrix.read_rows(runner.directory/'baseline.jsonl')
    baseline[0]['text']='changed evidence'
    matrix.write_rows(runner.directory/'baseline.jsonl',baseline)
    before=artifacts(runner.directory)
    runner.args.resume=True
    runner.fail_after=None
    with pytest.raises(ValueError,match='C0 baseline changed'):matrix.retrieval(runner.args)
    assert artifacts(runner.directory)==before


def test_resume_cannot_write_while_another_writer_holds_the_lock(runner):
    matrix.retrieval(runner.args)
    runner.args.resume=True
    before=artifacts(runner.directory)
    with FileLock(runner.directory/'retrieval.lock'):
        with pytest.raises(Timeout):matrix.retrieval(runner.args)
    assert artifacts(runner.directory)==before


@pytest.mark.parametrize('count',[0,1])
def test_paired_c4_c6_retrieval_never_constructs_a_reranker_and_resumes(runner,monkeypatch,count):
    runner.args.configurations=('C4','C6')
    original_reranker=matrix.CrossEncoderReranker
    def forbidden(*args,**kwargs):
        pytest.fail('the C4/C6-only acceptance run must not use the cross-encoder')
    monkeypatch.setattr(matrix,'CrossEncoderReranker',forbidden)
    runner.fail_after=count
    with pytest.raises(OSError):matrix.retrieval(runner.args)
    prefix=matrix.read_rows(runner.directory/'pairs.jsonl')
    runner.fail_after=None
    runner.args.resume=True
    matrix.retrieval(runner.args)
    pairs=matrix.read_rows(runner.directory/'pairs.jsonl')
    assert pairs[:count]==prefix
    assert all(list(t['rows'])==['C4','C6'] for t in pairs)
    assert len(pairs[0]['rows']['C6']['sub_results'])==2
    for key in ('retrieved_chunk_ids','retrieved_scores','latency_ms',*matrix.METRICS):
        assert pairs[1]['rows']['C6'][key]==pairs[1]['rows']['C4'][key]
    config=json.loads((runner.directory/'config.json').read_text())
    assert list(config['matrix'])==['C4','C6'] and config['reranker'] is None
    runtime=json.loads((runner.directory/'runtime.json').read_text())
    assert runtime['identity']['cross_encoder_device'] is None and not runtime['cross_encoder_warmed']
    assert not (runner.directory/'C5').exists()
    summaries=json.loads((runner.directory/'summary.json').read_text())['configurations']
    assert [s['config']['id'] for s in summaries]==['C4','C6']
    original=artifacts(runner.directory)
    matrix.retrieval(runner.args)
    assert artifacts(runner.directory)==original
    monkeypatch.setattr(matrix,'CrossEncoderReranker',original_reranker)
    runner.args.configurations=None
    with pytest.raises(ValueError,match='identical frozen measurement'):matrix.retrieval(runner.args)


def test_paired_retrieval_refuses_a_missing_frozen_row(runner):
    runner.args.configurations=('C4','C6')
    matrix.retrieval(runner.args)
    rows=matrix.read_rows(runner.directory/'pairs.jsonl')
    del rows[0]['rows']['C6']
    matrix.write_rows(runner.directory/'pairs.jsonl',rows)
    before=artifacts(runner.directory)
    runner.args.resume=True
    with pytest.raises(ValueError,match='every matrix row'):matrix.retrieval(runner.args)
    assert artifacts(runner.directory)==before


def answer_row(q):
    config=GenerationConfig('mistral','test',PROMPT_TEMPLATE_ID)
    answer=Answer(q.question,'Saved answer.',(),(),False,config,1.0)
    cost=query_cost('mistral',[],attempts=1,prices=None,successful_calls=1)
    return {'question_id':q.question_id,'source':q.source,'answer':answer.to_dict(),
            'quality':{'method':'judge_failed','error':'offline','faithfulness':None,'correctness':None,
                       'answerability_correct':1.0,'citation_precision':None,'citation_recall':0.0},
            'cost':cost,'latency_ms':{k:0.0 for k in ('retrieve','rerank','generate','retry_wait','total','other')},
            'judge_overhead':{'latency_ms':0.0,'retry_wait_ms':0.0,'total_ms':0.0,
                             'cost':query_cost('mistral',[],attempts=1,prices=None,successful_calls=0)}}


@pytest.fixture
def answer_run(runner,monkeypatch,request):
    if hasattr(request,'param'):
        monkeypatch.setitem(matrix.MATRIX,'C6',replace(matrix.MATRIX['C6'],**request.param))
    matrix.retrieval(runner.args)
    # Answer provenance shares the exact one-read snapshot just as retrieval does.
    monkeypatch.setattr(matrix,'FixedSizeBM25Retriever',runner.real_fixed)
    generate=importlib.import_module('src.rag.generate')
    quality=importlib.import_module('src.evaluation.quality')
    harness=importlib.import_module('src.evaluation.harness')
    generation=GenerationConfig('mistral','test',PROMPT_TEMPLATE_ID)
    monkeypatch.setattr(generate,'config_from_env',lambda **kwargs:generation)
    monkeypatch.setattr(generate,'checked_model',lambda *args:object())
    state=SimpleNamespace(generated=[],judged=[],stop=None)
    class Judge:
        config=generation
        def __init__(self,*args,**kwargs):pass
        def __call__(self,q,answer):
            state.judged.append(q.question_id)
            assert answer.text=='Saved answer.'
            return {'faithfulness':1.0,'correctness':.5,'faithfulness_reason':'supported',
                    'correctness_reason':'partial','input_tokens':10,'output_tokens':3}
    monkeypatch.setattr(quality,'LLMJudge',Judge)
    def evaluate(questions,retriever,config,**kwargs):
        q=questions[0]
        state.generated.append((q.question_id,kwargs))
        assert retriever.search(kwargs["stack"].scoped(matrix._query(q,top_k=16,metadata_filter=True))) is not None
        row=answer_row(q)
        report={'results':[row],'stopped':{'stage':'judge','question_id':q.question_id,'error':'offline'}}
        if state.stop:raise state.stop(report)
        return report
    monkeypatch.setattr(harness,'evaluate',evaluate)
    for cid in matrix.MATRIX:
        matrix.write_rows(runner.directory/cid/'answers.jsonl',[answer_row(q) for q in runner.questions])
    facts=runner.directory/'facts.parquet'
    facts.write_bytes(b'frozen facts')
    state.args=SimpleNamespace(run_dir=runner.directory,processed_dir=runner.args.processed_dir,
                               facts_file=facts,provider='mistral',model='test')
    state.runner=runner
    return state


def test_paired_retrieval_answers_and_rescore_use_the_frozen_row_selection(answer_run):
    state=answer_run
    directory=state.args.run_dir
    config=json.loads((directory/'config.json').read_text())
    config['matrix']={cid:config['matrix'][cid] for cid in ('C4','C6')}
    matrix.dump(directory/'config.json',config)
    pairs=matrix.read_rows(directory/'pairs.jsonl')
    for pair in pairs:pair['rows']={cid:pair['rows'][cid] for cid in ('C4','C6')}
    matrix.write_rows(directory/'pairs.jsonl',pairs)
    matrix.write_rows(directory/'C4/answers.jsonl',[])
    matrix.write_rows(directory/'C6/answers.jsonl',[])
    state.args.configurations=('C6',)
    matrix.answers(state.args)
    first=json.loads((directory/'answer-summary.json').read_text())
    assert not first['complete'] and list(first['configurations'])==['C6']
    state.args.configurations=None
    matrix.answers(state.args)
    assert len(state.generated)==4
    assert [kw['use_verification'] for _,kw in state.generated]==[True,True,False,False]
    summary=json.loads((directory/'answer-summary.json').read_text())
    assert summary['complete'] and list(summary['configurations'])==['C4','C6']
    matrix.rescore(state.args)
    rescored=json.loads((directory/'rescored/answer-summary.json').read_text())
    assert rescored==summary
    retrieval=json.loads((directory/'rescored/summary.json').read_text())
    assert [s['config']['id'] for s in retrieval['configurations']]==['C4','C6']
    state.args.configurations=('C5',)
    before=artifacts(directory)
    with pytest.raises(ValueError,match='no frozen retrieval'):matrix.answers(state.args)
    assert artifacts(directory)==before


@pytest.mark.parametrize('cid',['C0','C6'])
@pytest.mark.parametrize('stopped',[RunStopped,RunInterrupted])
def test_answer_loop_checkpoints_failed_judge_and_resumes_without_regeneration(answer_run,cid,stopped):
    state=answer_run
    directory=state.args.run_dir/cid
    matrix.write_rows(directory/'answers.jsonl',[])
    state.stop=stopped
    with pytest.raises(stopped):matrix.answers(state.args)
    assert matrix.read_rows(directory/'answers.jsonl')==[]
    pending=json.loads((directory/'pending-judge.json').read_text())
    assert pending['question_id']=='q1'
    state.stop=None
    matrix.answers(state.args)
    rows=matrix.read_rows(directory/'answers.jsonl')
    assert [r['question_id'] for r in rows]==['q1','q2']
    assert rows[0]['judge_resumed'] is True and rows[0]['answer']==pending['answer']
    assert rows[0]['judge_overhead']['cost']['model_calls']==2
    assert [qid for qid,_ in state.generated]==['q1','q2']
    assert state.judged==['q1'] and not (directory/'pending-judge.json').exists()
    flags=state.generated[0][1]
    assert flags['use_verification']==matrix.MATRIX[cid].use_verification
    assert flags['use_decomposition']==matrix.MATRIX[cid].use_decomposition
    assert rows[0]['evidence_rerank_enabled']==matrix.MATRIX[cid].rerank


def test_answer_loop_clears_a_stale_checkpoint_even_when_the_row_is_already_complete(answer_run):
    state=answer_run
    pending=state.args.run_dir/'C0/pending-judge.json'
    matrix.dump(pending,answer_row(state.runner.questions[0]))
    matrix.answers(state.args)
    assert not pending.exists() and not state.generated and not state.judged


def test_soclaas_answers_can_use_a_separate_frozen_mistral_judge(answer_run,monkeypatch):
    state=answer_run
    generation_module=importlib.import_module('src.rag.generate')
    quality=importlib.import_module('src.evaluation.quality')
    configs={
        'soclaas':GenerationConfig('soclaas','qwen3.6:35b',PROMPT_TEMPLATE_ID),
        'mistral':GenerationConfig('mistral','ministral-8b-2512',PROMPT_TEMPLATE_ID),
    }
    clients={provider:object() for provider in configs}
    monkeypatch.setattr(generation_module,'config_from_env',lambda *,provider,model:replace(
        configs[provider],model=model or configs[provider].model))
    monkeypatch.setattr(generation_module,'checked_model',lambda config,llm:clients[config.provider])
    class Judge:
        def __init__(self,config,*,llm):
            self.config=config
            assert config==configs['mistral'] and llm is clients['mistral']
    monkeypatch.setattr(quality,'LLMJudge',Judge)
    state.args.provider='soclaas'
    state.args.model='qwen3.6:35b'
    state.args.judge_provider='mistral'
    state.args.judge_model='ministral-8b-2512'
    state.args.configurations=('C4',)
    matrix.write_rows(state.args.run_dir/'C4/answers.jsonl',[])
    matrix.answers(state.args)
    config=json.loads((state.args.run_dir/'answers-config.json').read_text())
    assert config['generation']==configs['soclaas'].to_dict()
    assert config['judge_generation']==configs['mistral'].to_dict()
    assert all(kw['llm'] is clients['soclaas'] and kw['judge'].config==configs['mistral']
               for _,kw in state.generated)
    before=artifacts(state.args.run_dir)
    state.args.judge_model='different-judge'
    with pytest.raises(ValueError,match='identical settings/source/facts'):
        matrix.answers(state.args)
    assert artifacts(state.args.run_dir)==before


def test_answers_default_judge_remains_the_generator(answer_run):
    matrix.answers(answer_run.args)
    config=json.loads((answer_run.args.run_dir/'answers-config.json').read_text())
    assert config['generation']==config['judge_generation']


def test_answer_loop_can_measure_only_the_rebased_c6_row(answer_run):
    state=answer_run
    state.args.configurations=("C6",)
    matrix.write_rows(state.args.run_dir/'C6/answers.jsonl',[])
    matrix.answers(state.args)
    assert [qid for qid,_ in state.generated]==['q1','q2']
    summary=json.loads((state.args.run_dir/'answer-summary.json').read_text())
    assert summary['complete'] is False
    assert list(summary['configurations'])==['C6']


@pytest.mark.parametrize('first,then',[(None,('C6',)),(('C6',),None)])
def test_answer_loop_adds_rows_to_a_run_that_already_answered_others(answer_run,first,then):
    state=answer_run
    matrix.write_rows(state.args.run_dir/'C6/answers.jsonl',[])
    state.args.configurations=first
    matrix.answers(state.args)
    state.args.configurations=then
    matrix.answers(state.args)
    assert [qid for qid,_ in state.generated]==['q1','q2']
    summary=json.loads((state.args.run_dir/'answer-summary.json').read_text())
    assert summary['complete'] is True and list(summary['configurations'])==list(matrix.MATRIX)


def test_answer_loop_refuses_a_repeated_row(answer_run):
    answer_run.args.configurations=('C6','C6')
    with pytest.raises(ValueError,match='unique matrix row IDs'):matrix.answers(answer_run.args)
    assert not answer_run.generated


def test_answer_loop_rejects_a_nonprefix_before_generation_or_file_rewrite(answer_run):
    state=answer_run
    path=state.args.run_dir/'C0/answers.jsonl'
    matrix.write_rows(path,[answer_row(state.runner.questions[1])])
    before=path.read_bytes()
    with pytest.raises(ValueError,match='exact team prefix'):matrix.answers(state.args)
    assert path.read_bytes()==before and not state.generated


def test_answer_replay_rejects_changed_c0_text_before_models_are_created(answer_run):
    state=answer_run
    baseline=matrix.read_rows(state.args.run_dir/'baseline.jsonl')
    baseline[0]['text']='changed C0 text'
    matrix.write_rows(state.args.run_dir/'baseline.jsonl',baseline)
    before=artifacts(state.args.run_dir)
    with pytest.raises(ValueError,match='C0 baseline changed'):matrix.answers(state.args)
    assert artifacts(state.args.run_dir)==before and not state.generated


def test_answer_resume_allows_os_path_and_workload_changes_but_records_them(answer_run):
    state=answer_run
    matrix.answers(state.args)
    frozen=(state.args.run_dir/'answers-config.json').read_bytes()
    state.runner.platform='different OS build'
    state.args.processed_dir=Path('another-spelling-of-same-snapshot')
    state.args.workload_note='resumed on the same machine overnight'
    matrix.answers(state.args)
    assert (state.args.run_dir/'answers-config.json').read_bytes()==frozen
    runtime=json.loads((state.args.run_dir/'answers-runtime-0001.json').read_text())
    assert runtime['source']['environment']['platform']=='different OS build'
    assert runtime['source']['corpus']['processed_dir']==str(state.args.processed_dir)
    assert runtime['workload_note']==state.args.workload_note
    assert runtime['configurations']==list(matrix.MATRIX)


@pytest.mark.parametrize('answer_run',[
    {'use_verification':False,'use_decomposition':False,'rerank':False}],indirect=True)
def test_answer_flags_follow_the_registry_instead_of_the_c6_id(answer_run):
    state=answer_run
    matrix.write_rows(state.args.run_dir/'C6/answers.jsonl',[])
    matrix.answers(state.args)
    assert all(not flags['use_verification'] and not flags['use_decomposition'] for _,flags in state.generated)
    summary=json.loads((state.args.run_dir/'C6/answer-summary.json').read_text())
    assert summary['verification_enabled'] is False
    assert all(not r['evidence_rerank_enabled'] for r in matrix.read_rows(state.args.run_dir/'C6/answers.jsonl'))


@pytest.mark.parametrize('argv',[
    ['retrieval','questions.jsonl','--run-id','x','--workload-note','test','--reference','reference'],
    ['retrieval','questions.jsonl','--run-id','x','--workload-note','test','--reference-pairs-sha256','a'*64],
])
def test_cli_requires_reference_and_archived_digest_together(argv):
    with pytest.raises(SystemExit) as stopped:matrix.main(argv)
    assert stopped.value.code==2


def test_cli_accepts_soclaas_and_separate_judge_options(monkeypatch):
    seen=[]
    monkeypatch.setattr(matrix,'answers',lambda args:seen.append(args))
    matrix.main(['answers','results/example','--provider','soclaas','--model','qwen3.6:35b',
                 '--judge-provider','mistral','--judge-model','ministral-8b-2512'])
    assert seen[0].provider=='soclaas' and seen[0].model=='qwen3.6:35b'
    assert seen[0].judge_provider=='mistral' and seen[0].judge_model=='ministral-8b-2512'
