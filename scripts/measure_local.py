"""Measure parser memory and retrieval latency without hosted services."""
import concurrent.futures
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("DOCMIND_HOME",str(ROOT / "reports" / "ops_workspace"))
import numpy as np
from docmind.ingestion.loader import load_pdf
from docmind.retrieval.sparse import BM25Index,tokenize


def rss():
    class Counters(ctypes.Structure):
        _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
          ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
          ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),
          ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),
          ("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
    if os.name != "nt":
        import resource
        return {"peak_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    counter=Counters();counter.cb=ctypes.sizeof(counter)
    process=ctypes.windll.kernel32.GetCurrentProcess
    process.restype=wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo(process(),ctypes.byref(counter),counter.cb)
    return {"rss_mib":counter.WorkingSetSize/2**20,"peak_mib":counter.PeakWorkingSetSize/2**20}


def pdf(path,n):
    # Valid synthetic text PDF with independent page objects and a shared font.
    objs=[b"",b"",b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    pages=[]
    text="".join(f"BT /F1 12 Tf 50 {750-line*20} Td (Synthetic parser memory fixture line {line}. Coolant stays below 300 degrees.) Tj ET\n" for line in range(25)).encode()
    for i in range(n):
        page=len(objs)+1;content=page+1;pages.append(page)
        objs.extend([f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content} 0 R /Resources << /Font << /F1 3 0 R >> >> >>".encode(),
          b"<< /Length %d >>\nstream\n"%len(text)+text+b"endstream"])
    objs[0]=b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[1]=f"<< /Type /Pages /Kids [{' '.join(str(p)+' 0 R' for p in pages)}] /Count {n} >>".encode()
    data=b"%PDF-1.4\n";offsets=[]
    for i,obj in enumerate(objs,1):
        offsets.append(len(data));data+=f"{i} 0 obj\n".encode()+obj+b"\nendobj\n"
    start=len(data)
    data+=f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    data+=b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    data+=f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    path.write_bytes(data)


def main():
    report={"python":platform.python_version(),"cpu_count":os.cpu_count(),"scope":"synthetic parser and BM25 microbenchmarks, not model/API load"}
    report["parser"]=[]
    with tempfile.TemporaryDirectory() as tmp:
        for n in (100,300):
            path=Path(tmp)/f"{n}.pdf";pdf(path,n)
            before=rss();start=time.perf_counter();docs=load_pdf(path);seconds=time.perf_counter()-start
            report["parser"].append({"pages":n,"bytes":path.stat().st_size,"seconds":seconds,
                "nonempty_pages":len(docs),"before":before,"after":rss()})
    corpus=[tokenize(f"document {i} coolant pump temperature protocol topic{i%100}") for i in range(50000)]
    index=BM25Index();start=time.perf_counter();index.fit(corpus)
    report["bm25_build_seconds"]=time.perf_counter()-start
    def query(i):
        start=time.perf_counter();scores=index.get_scores(tokenize(f"coolant topic{i%100}"))
        np.argpartition(-scores,4)[:5]
        return (time.perf_counter()-start)*1000
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        timings=list(pool.map(query,range(400)))
    report["bm25_load"]={"queries":400,"workers":4,"chunks":50000,"wall_seconds":time.perf_counter()-start,
        "p50_ms":float(np.percentile(timings,50)),"p95_ms":float(np.percentile(timings,95)),"errors":0}
    out=ROOT/"reports"/"local_ops.json";out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=="__main__":main()
