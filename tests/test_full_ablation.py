"""Controlled matrix transitions, recorded evidence, and completion guards."""
import json
import hashlib
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from src.evaluation import full_ablation as matrix
from src.evaluation.records import BenchmarkQuestion
from src.retrieval.records import Query, RetrievedPassage
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig
from conftest import RerankModel


def question(identifier="q", source="handwritten"):
    return BenchmarkQuestion(identifier, "Compare Apple and Microsoft's revenue in FY2024",
                             "Apple and Microsoft revenue", ("a", "b"), (), None, 2024,
                             "comparative", "hard", source)


def chunk(identifier, ticker):
    return {"chunk_id":identifier,"text":"Revenue | 2024 | 100", "ticker":ticker,
            "company":ticker,"fiscal_year":2024,"item":"8","title":"Statements",
            "url":"https://example.test","content_type":"table","accession_no":ticker}


class Part:
    name="bm25"

    def __init__(self, chunks):
        self.chunks,self.calls=chunks,[]

    def search(self, query, k=None):
        self.calls.append((query,k))
        kept=[c for c in self.chunks if not query.tickers or c["ticker"] in query.tickers]
        return [RetrievedPassage.from_chunk(c,1.0,i,self.name) for i,c in enumerate(kept,1)][:k]

    def relevant_chunk_ids(self, q, hard=False):
        return list(q.hard_negative_chunk_ids if hard else q.supporting_chunk_ids)


def test_matrix_filters_are_the_only_c3_c4_query_change():
    q=Query("Revenue?",tickers=("AAPL",),fiscal_years=(2024,),items=("8",),
            keyword_text="revenue",table_boost=2.0,wants_figures=True,top_k=16)
    raw=matrix.MATRIX["C3"].scoped(q)
    assert raw==replace(q,tickers=(),fiscal_years=(),items=(),content_type=None)
    assert matrix.MATRIX["C4"].scoped(q)==q
    for cid, config in matrix.MATRIX.items():
        assert not config.use_facts and not config.use_refusal
        assert config.use_decomposition==(cid=="C6")


def test_matrix_shares_components_and_c6_reranks_each_filing():
    from src.retrieval.hybrid import HybridRetriever
    chunks=[chunk("a","AAPL"),chunk("b","MSFT")]
    sparse,dense=Part(chunks),Part(chunks);dense.name="dense"
    hybrid=HybridRetriever(sparse,dense)
    reranker=CrossEncoderReranker(RerankConfig(),model=RerankModel([1.0,2.0]))
    # One-candidate sub-pools need a model with one score per supplied pair.
    reranker._model.predict=lambda pairs,**kwargs: [float(i) for i in range(len(pairs))]
    result=matrix.measure_question(question(),Part(chunks),hybrid,reranker,{c['chunk_id']:c for c in chunks})
    rows=result['rows'];assert list(rows)==list(matrix.MATRIX)
    assert rows['C3']['query']['keyword_text']==rows['C4']['query']['keyword_text']
    assert rows['C3']['query']['table_boost']==rows['C4']['query']['table_boost']
    assert rows['C3']['query']['tickers']==() and rows['C4']['query']['tickers']==('AAPL','MSFT')
    assert len(rows['C6']['sub_results'])==2
    assert set(rows['C6']['retrieved_chunk_ids'])=={'a','b'}
    assert len([q for q,k in sparse.calls if not q.tickers])==1
    saved=json.loads(json.dumps(rows['C6']))
    replay=matrix.ReplayRetriever(saved,{c['chunk_id']:c for c in chunks})
    for sub in saved['sub_results']:
        q=Query(**{**sub['query'],'tickers':tuple(sub['query']['tickers']),
                   'fiscal_years':tuple(sub['query']['fiscal_years']),'items':tuple(sub['query']['items'])})
        assert [p.chunk_id for p in replay.search(q)]==sub['retrieved_chunk_ids']
    with pytest.raises(ValueError,match="differs"):
        replay.search(Query('different'))


def test_single_filing_c6_is_exactly_c5():
    from src.retrieval.hybrid import HybridRetriever
    chunks=[chunk('a','AAPL')];part=Part(chunks)
    q=replace(question(),question='What was Apple revenue in FY2024?',ticker='AAPL',supporting_chunk_ids=('a',))
    rows=matrix.measure_question(q,part,HybridRetriever(part,Part(chunks)),
         CrossEncoderReranker(RerankConfig(),model=RerankModel([1.0])),{'a':chunks[0]})['rows']
    assert rows['C6']['origin']=='C5_identical_no_split'
    for field in ('retrieved_chunk_ids','retrieved_scores','latency_ms',*matrix.METRICS):
        assert rows['C6'][field]==rows['C5'][field]


def test_transaction_reader_recovers_only_the_unfinished_tail(tmp_path):
    path=tmp_path/'pairs.jsonl';path.write_bytes(b'{"complete":true}\n{"partial":')
    with pytest.raises(EOFError):matrix.read_rows(path)
    assert matrix.read_rows(path,recover=True)==[{'complete':True}]
    path.write_bytes(b'{"broken":\n')
    with pytest.raises(json.JSONDecodeError):matrix.read_rows(path,recover=True)


def test_summary_exposes_denominators_and_sources(tmp_path):
    q=question(); rows={}
    for cid in matrix.MATRIX:
        rows[cid]=matrix.ranking(q,Query(q.question),[],cid,10.0)
    matrix.export_retrieval(tmp_path,[{'question':q.to_dict(),'rows':rows}],1)
    report=json.loads((tmp_path/'summary.json').read_text())
    assert report['complete'] and len(report['configurations'])==7
    assert report['configurations'][0]['metric_denominators']['recall']==1
    assert report['configurations'][0]['metric_denominators']['hard_negative_accuracy']==0
    assert json.loads((tmp_path/'by-source/summary.json').read_text())['configurations'][0]['source']=='handwritten'
    with pytest.raises(ValueError,match='incomplete'):
        matrix.export_retrieval(tmp_path,[],1)


def test_reference_rejects_mismatched_corpus_before_reading_pairs(tmp_path):
    reranker=CrossEncoderReranker(RerankConfig(),model=RerankModel([1.0]))
    provenance={'question_records_sha256':'q','retrieval_settings':{},'corpus':{'records_sha256':'new'}}
    original={**provenance,'corpus':{'records_sha256':'old'}}
    matrix.dump(tmp_path/'config.json',{'provenance':original})
    matrix.dump(tmp_path/'runtime.json',{})
    with pytest.raises(ValueError,match='corpus'):
        matrix.reference_rows(tmp_path,[question()],provenance,SimpleNamespace(),reranker)


def test_verification_switch_is_an_actual_answer_stage_ablation(monkeypatch,tmp_path):
    from src.evaluation import harness
    from src.rag.records import Answer,GenerationConfig,VerificationResult
    from src.rag.constants import PROMPT_TEMPLATE_ID
    config=GenerationConfig('ollama','test',PROMPT_TEMPLATE_ID)
    q=question();answer=Answer(q.question,'Revenue increased.',(),(),False,config,1.0)
    monkeypatch.setattr(harness,'answer_question',lambda *args,**kwargs:answer)
    calls=[]
    def verify(result,**kwargs):
        calls.append(kwargs)
        return replace(result,verification=VerificationResult(q.question_type,()))
    monkeypatch.setattr(harness,'verify_answer',verify)
    facts=tmp_path/'facts.parquet'
    off=harness.evaluate([q],Part([]),config,run_id='off',use_verification=False,facts_file=facts)
    assert off['results'][0]['answer']['verification'] is None
    assert off['checks']=={'not_run':1} and not calls
    on=harness.evaluate([q],Part([]),config,run_id='on',use_verification=True,facts_file=facts)
    assert on['checks']=={'unchecked':1} and calls[0]['facts_file']==facts
    assert on['results'][0]['answer']['text']==off['results'][0]['answer']['text']


def test_reference_requires_exact_queries_and_complete_question_sequence(tmp_path):
    import gzip
    from src.evaluation.run import _query
    q=question();reranker=CrossEncoderReranker(RerankConfig(),model=RerankModel([1.0]))
    provenance={'question_records_sha256':'q','retrieval_settings':{},'corpus':{'records_sha256':'same'},'source_commit':'frozen'}
    matrix.dump(tmp_path/'config.json',{'provenance':provenance,'top_k':16,'reranker':reranker.config.to_dict()})
    matrix.dump(tmp_path/'runtime.json',{'bm25_manifest':None,'dense_manifest':None})
    record={'question':q.to_dict(),'query':json.loads(json.dumps(asdict(_query(q,top_k=16,metadata_filter=True))))}
    def save(rows):
        with gzip.open(tmp_path/'pairs.jsonl.gz','wt') as stream:
            stream.writelines(json.dumps(r)+'\n' for r in rows)
    save([record]);rows,meta=matrix.reference_rows(tmp_path,[q],provenance,SimpleNamespace(),reranker)
    assert rows==[record] and meta['source_commit']=='frozen'
    save([])
    with pytest.raises(ValueError,match='complete'):
        matrix.reference_rows(tmp_path,[q],provenance,SimpleNamespace(),reranker)
    record['query']['keyword_text']='different';save([record])
    with pytest.raises(ValueError,match='query policy'):
        matrix.reference_rows(tmp_path,[q],provenance,SimpleNamespace(),reranker)


def test_pending_judge_keeps_the_completed_answer_and_all_attempt_costs(tmp_path):
    from src.rag.records import Answer,GenerationConfig,Citation,SentenceCitations
    from src.rag.constants import PROMPT_TEMPLATE_ID
    from src.rag.generate import ProviderUnavailable
    from src.evaluation.quality import query_cost
    config=GenerationConfig('mistral','test',PROMPT_TEMPLATE_ID);q=question()
    p=RetrievedPassage.from_chunk(chunk('a','AAPL'),1.0,1,'hybrid')
    citation=Citation(1,'a',True)
    answer=Answer(q.question,'Revenue was 100. [1]',(citation,),(p,),False,config,12.0,
                  sentences=(SentenceCitations('Revenue was 100. [1]','Revenue was 100.',(citation,)),))
    assert matrix.saved_answer(answer.to_dict()).to_dict()==answer.to_dict()
    path=tmp_path/'pending.json'
    row={'question_id':q.question_id,'answer':answer.to_dict(),'quality':{'method':'judge_failed','error':'old'},
         'judge_overhead':{'latency_ms':0.0,'retry_wait_ms':0.0,'total_ms':0.0,
                          'cost':query_cost('mistral',[],attempts=1,prices=None,successful_calls=0)}}
    matrix.dump(path,row)
    class Judge:
        def __init__(self):self.config=config;self.fail=True
        def __call__(self,question,saved):
            assert saved.to_dict()==answer.to_dict()
            if self.fail:raise ProviderUnavailable('offline')
            return {'faithfulness':1.0,'correctness':.5,'faithfulness_reason':'supported',
                    'correctness_reason':'partial','input_tokens':10,'output_tokens':3}
    judge=Judge()
    with pytest.raises(ProviderUnavailable):matrix.finish_pending_judge(path,q,judge)
    assert json.loads(path.read_text())['judge_overhead']['cost']['model_calls']==2
    judge.fail=False;completed=matrix.finish_pending_judge(path,q,judge)
    assert completed['answer']==answer.to_dict() and completed['quality']['method']=='llm_judge'
    cost=completed['judge_overhead']['cost']
    assert cost['model_calls']==3 and cost['successful_calls']==1 and cost['usage_calls']==1
    assert cost['observed_input_tokens']==10 and cost['input_tokens'] is None and cost['usd'] is None


def test_transactions_require_every_row_and_preserve_the_benchmark_order():
    q=question();t={'question':q.to_dict(),'rows':{}}
    with pytest.raises(ValueError,match='every matrix row'):matrix.check_transactions([t],[q])
    with pytest.raises(ValueError,match='exact benchmark prefix'):matrix.check_transactions([t],[replace(q,question_id='other')])


def test_rescore_recomputes_metrics_at_the_recorded_cutoff_without_models(tmp_path,monkeypatch):
    from src.evaluation.quality import query_cost
    q=question();passages=[RetrievedPassage.from_chunk(chunk('a','AAPL'),1.0,1,'bm25')]
    rows={cid:matrix.ranking(q,Query(q.question),passages,cid,12.0) for cid in matrix.MATRIX}
    for row in rows.values():row['recall']=999.0
    digest=hashlib.sha256(json.dumps([q.to_dict()],sort_keys=True).encode()).hexdigest()
    matrix.dump(tmp_path/'config.json',{'provenance':{'questions':1,'question_records_sha256':digest}})
    matrix.write_rows(tmp_path/'benchmark.jsonl',[q.to_dict()])
    matrix.write_rows(tmp_path/'pairs.jsonl',[{'question':q.to_dict(),'rows':rows}])
    cost=query_cost('ollama',[],attempts=0,prices=None)
    answer={'question_id':q.question_id,'source':q.source,
            'quality':{'faithfulness':None,'correctness':0.0,'answerability_correct':0.0,
                       'citation_precision':None,'citation_recall':0.0},
            'answer':{'abstained':True,'verification':None},'cost':cost,
            'latency_ms':{k:0.0 for k in ('retrieve','rerank','generate','retry_wait','total','other')},
            'judge_overhead':{'latency_ms':0.0,'retry_wait_ms':0.0,'total_ms':0.0,'cost':cost}}
    for cid in matrix.MATRIX:
        (tmp_path/cid).mkdir();matrix.write_rows(tmp_path/cid/'answers.jsonl',[answer])
    monkeypatch.setattr(matrix,'FINAL_K',1)
    matrix.rescore(SimpleNamespace(run_dir=tmp_path))
    result=json.loads((tmp_path/'rescored/summary.json').read_text())
    assert result['top_k']==16 and result['configurations'][0]['recall']==.5
    assert result['configurations'][0]['ndcg']<1.0
    assert json.loads((tmp_path/'rescored/answer-summary.json').read_text())['team_questions']==1
    (tmp_path/'C6/answers.jsonl').unlink()
    with pytest.raises(ValueError,match='incomplete C6'):matrix.rescore(SimpleNamespace(run_dir=tmp_path))


def test_snapshot_cannot_change_together_with_its_recorded_rankings():
    q=question();digest=hashlib.sha256(json.dumps([q.to_dict()],sort_keys=True).encode()).hexdigest()
    config={'provenance':{'questions':1,'question_records_sha256':digest}}
    matrix.validate_snapshot(config,[q])
    with pytest.raises(ValueError,match='frozen manifest'):
        matrix.validate_snapshot(config,[replace(q,question='An edited benchmark question')])
