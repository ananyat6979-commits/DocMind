"""Thin UI. The authenticated service owns queries, library mutations and review state."""
import json
import os
import urllib.error
import urllib.request
import uuid


def render():
    import streamlit as st
    base=os.getenv("DOCMIND_API_URL","http://127.0.0.1:8000").rstrip("/")
    st.title("DocMind")
    token=st.sidebar.text_input("Personal access key",type="password",key="personal_access_key")
    if not token:
        st.info("Enter your access key to open the document library.")
        return
    def request(method,path,body=None,upload=None):
        headers={"X-API-Key":token}
        data=None
        if upload is not None:
            boundary=uuid.uuid4().hex
            name=upload.name.replace('"',"_").replace("\r","_").replace("\n","_")
            content=upload.getvalue()
            if len(content)>50*1024*1024: raise ValueError("File exceeds 50 MiB")
            data=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()+content+f'\r\n--{boundary}--\r\n'.encode())
            headers["Content-Type"]="multipart/form-data; boundary="+boundary
        elif body is not None:
            data=json.dumps(body).encode();headers["Content-Type"]="application/json"
        req=urllib.request.Request(base+path,data=data,headers=headers,method=method)
        try:
            with urllib.request.urlopen(req,timeout=180) as response:return json.load(response)
        except urllib.error.HTTPError as exc:
            try: detail=json.load(exc).get("detail","Request rejected")
            except (ValueError,AttributeError): detail="Request rejected"
            raise ValueError(str(detail)) from exc
        except OSError as exc:raise ValueError("The document service is unavailable") from exc
    try: principal=request("GET","/whoami")
    except ValueError as exc:
        st.error(str(exc));return
    st.sidebar.caption("Signed in as "+principal["actor_id"])
    if principal.get("generation_mode") == "offline_extractive_demo":
        st.info("Offline workflow demonstration. The extractive reader does not measure generative answer quality.")
    if st.session_state.get("response_actor")!=principal["actor_id"]:
        st.session_state.response_ids=[];st.session_state.response_actor=principal["actor_id"]
    roles=principal["roles"]
    try:
        documents=request("GET","/documents")
        for document in documents:
            if document.get("lineage",{}).get("partial_document"):
                st.warning(document["filename"]+": some pages were quarantined; answers cover readable pages only.")
        if "ingestor" in roles:
            uploaded=st.sidebar.file_uploader("Add a document",type=["pdf","docx","txt","md"])
            if uploaded is not None and st.sidebar.button("Upload document"):
                job=request("POST","/uploads/jobs",upload=uploaded)
                st.session_state.ingestion_job=job["job_id"]
            if st.session_state.get("ingestion_job") and st.sidebar.button("Check upload status"):
                job=request("GET","/ingest/jobs/"+st.session_state.ingestion_job)
                st.sidebar.write(job["status"])
                if job.get("error_detail"):st.sidebar.warning(job["error_detail"].get("action", job["error_detail"].get("message")))
                if job.get("result"):
                    for item in json.loads(job["result"]).get("files", []):
                        st.sidebar.write(item["filename"] + ": " + item["status"])
                        if item.get("stored_filename") != item["filename"]:
                            st.sidebar.caption("Cached in library as " + item.get("stored_filename", ""))
                        missing=item.get("quality", {}).get("quarantined_pages", [])
                        if missing:st.sidebar.warning("Quarantined pages: " + ", ".join(map(str, missing)))
        if "admin" in roles and documents:
            chosen=st.sidebar.selectbox("Document to remove",documents,format_func=lambda d:d["filename"])
            if st.sidebar.button("Remove document"):
                request("DELETE","/documents/"+chosen["file_id"]);st.rerun()
        options={d["file_id"]:d["filename"] for d in documents}
        selected=st.multiselect("Search documents",list(options),format_func=lambda value:options[value])
        with st.form("query"):
            question=st.text_input("Ask a question",max_chars=2000)
            submitted=st.form_submit_button("Ask")
        if submitted and question.strip():
            result=request("POST","/query",{"question":question,"file_ids":selected or None})
            st.session_state.response_ids.append(result["response_id"])
        for response_id in reversed(st.session_state.get("response_ids",[])):
            try: result=request("GET","/responses/"+response_id)
            except ValueError:
                st.info("This response is no longer available.");continue
            st.text(result["question"])
            st.text(result["answer"])
            st.caption(result["review_status"])
            if result.get("release_state") == "draft":
                st.caption("Unverified draft. Citation found; automated support has not been established.")
            for document in result.get("document_context", {}).get("documents", []):
                if document.get("quarantined_pages"):
                    st.warning(document["filename"] + ": unreadable pages " + ", ".join(map(str, document["quarantined_pages"])))
            for warning in result.get("warnings", []):
                st.warning(warning.replace("_", " "))
            for evidence in result.get("evidence", []):
                if evidence.get("status") in ("literal", "normalized"):
                    st.text(evidence.get("quote", evidence.get("text", "")))
                    st.caption(str(evidence.get("filename", "")) + " | page " + str(evidence.get("page", "")))
            for source in result["sources"]:
                with st.expander(source["filename"]):st.text(source["content_preview"])
            if st.button("Refresh response",key="refresh_"+response_id):st.rerun()
        if "reviewer" in roles:
            st.subheader("Human review")
            for item in request("GET","/review/pending"):
                rid=item["response_id"]
                with st.form("review_"+rid):
                    st.text(item["question"]);st.text(item["answer"])
                    for source in item["sources"]:st.text(source["content"])
                    decision=st.selectbox("Decision",["approved","edited","rejected"])
                    answer=st.text_area("Corrected answer")
                    note=st.text_input("Review note")
                    if st.form_submit_button("Save decision"):
                        request("POST","/review/"+rid,{"decision":decision,"edited_answer":answer or None,"note":note});st.rerun()
    except ValueError as exc:st.error(str(exc))
