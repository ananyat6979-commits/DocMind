"""Candidate sampling must preserve required labels and be independent of order."""
import gzip
import hashlib
import io
from scripts.prepare_miracl_component import CandidateSampler,HashedStream,_DOCID


def test_required_passages_survive_bounded_sampling():
    records=[{"docid":str(i),"text":"passage "+str(i)} for i in range(100)]
    sampler=CandidateSampler("hi",{"99"},3)
    for record in records:sampler.offer(record["docid"],lambda record=record:record)
    assert sampler.found["99"]["text"]=="passage 99"
    assert len(sampler.sample)==3 and len(sampler.documents())<=4
    reverse=CandidateSampler("hi",{"99"},3)
    for record in reversed(records):reverse.offer(record["docid"],lambda record=record:record)
    assert sampler.documents()==reverse.documents()


def test_hashing_covers_the_entire_compressed_source():
    content=b'{"docid": "123#0", "text": "passage"}\n'
    payload=gzip.compress(content)
    stream=HashedStream(io.BytesIO(payload))
    with gzip.GzipFile(fileobj=stream) as source:assert source.read()==content
    while stream.read(65536):pass
    assert stream.hash.hexdigest()==hashlib.sha256(payload).hexdigest()
    assert _DOCID.search(content).group(1)==b"123#0"
