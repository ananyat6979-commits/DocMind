"""Compare trusted native parsing with the isolated Unix-socket transport."""
import argparse
import os
import json
import hashlib
import urllib.request
from pathlib import Path
import tempfile
from docmind.evaluation.release import digest,semantic_documents
from docmind.ingestion.loader import load_document
from docmind.ingestion.parser_client import parse_remote


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--socket",required=True);parser.add_argument("--manifest",type=Path);args=parser.parse_args()
    os.environ.update(DOCMIND_PARSER_SOCKET=args.socket,DOCMIND_PARSER_MODE="local",DOCMIND_ALLOW_UNSANDBOXED_PARSING="1")
    from reportlab.pdfgen import canvas
    from docx import Document
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        pdf=root/"sample.pdf";drawing=canvas.Canvas(str(pdf));drawing.drawString(72,720,"The measured limit is 12.5 kg.");drawing.showPage();drawing.showPage();drawing.drawString(72,720,"Wrapped evidence remains available.");drawing.save()
        docx=root/"sample.docx";document=Document();document.add_heading("Procedure",1);document.add_paragraph("The measured limit is 12.5 kg.")
        table=document.add_table(rows=2,cols=2);table.cell(0,0).merge(table.cell(0,1)).text="Merged heading";table.cell(1,0).text="Limit";table.cell(1,1).text="12.5 kg";document.save(docx)
        for path in (pdf,docx):
            native=load_document(path,expected_language="en")
            isolated=parse_remote(path,"en")
            assert semantic_documents(native)==semantic_documents(isolated),"Parser transport changed model-consumed output"
            print(path.suffix,"semantic output equivalence",digest(semantic_documents(native)))
        bad=root/"bad.pdf";bad.write_bytes(b"%PDF-1.4 corrupt")
        try:parse_remote(bad)
        except ValueError:pass
        else:raise AssertionError("Corrupt PDF was accepted")
        assert parse_remote(pdf,"en"),"Service did not recover after corrupt input"
        if args.manifest:
            for source in json.loads(args.manifest.read_text(encoding="utf-8"))["sources"]:
                path=root/Path(source["filename"]).name
                request=urllib.request.Request(source["url"],headers={"User-Agent":"DocMindResearch/1.0 (+https://github.com/ananyat6979-commits/DocMind)"})
                with urllib.request.urlopen(request,timeout=60) as response,path.open("wb") as output:
                    size=0
                    while block:=response.read(65536):
                        size+=len(block)
                        if size>50*1024*1024:raise ValueError("Benchmark input exceeds size limit")
                        output.write(block)
                assert hashlib.sha256(path.read_bytes()).hexdigest()==source["pdf_sha256"],"Benchmark source changed"
                native=load_document(path,expected_language=source["expected_language"])
                isolated=parse_remote(path,source["expected_language"])
                native_hash=digest(semantic_documents(native));isolated_hash=digest(semantic_documents(isolated))
                assert native_hash==isolated_hash,"Actual benchmark transport changed extraction"
                assert isolated_hash==source["expected_semantic_sha256"],"Cached benchmark extraction differs: "+isolated_hash
                print("Cached benchmark and isolated output match",isolated_hash)
        print("Socket transport, rejection, recovery and output equivalence passed")

if __name__=="__main__":main()
