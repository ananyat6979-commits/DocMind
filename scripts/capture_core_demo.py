"""Capture the actual UI with synthetic sources, real E5 retrieval and extractive demo generation.

No hosted inference, human labels or real user credentials are used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def serve(out):
    with tempfile.TemporaryDirectory(prefix='docmind-demo-') as temporary:
        home=Path(temporary);registry=home/'credentials.json';keys={name:secrets.token_hex(24) for name in ('reader','reviewer')}
        registry.write_text(json.dumps([{'tenant_id':'demo','actor_id':'demo-'+name,'key_sha256':hashlib.sha256(key.encode()).hexdigest(),
            'roles':['reader','ingestor'] if name=='reader' else ['reader','reviewer']} for name,key in keys.items()]))
        os.environ.pop('GROQ_API_KEY',None)
        os.environ.update(DOCMIND_HOME=str(home),DOCMIND_CREDENTIALS_FILE=str(registry),DOCMIND_SECRETS_FILE=str(home/'absent.env'),
            DOCMIND_ATTRIBUTION='0',DOCMIND_REVIEW_AUDIT_RATE='0',DOCMIND_DEVICE='cpu',DOCMIND_TEST_MODE='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            DOCMIND_API_URL='http://127.0.0.1:8787',HF_HUB_OFFLINE='1')
        from docmind.config import CONFIG
        CONFIG.llm.provider='demo'
        assert not CONFIG.llm.groq_api_key and CONFIG.embedding.backend=='sentence-transformers'
        from docmind.ingestion.pipeline import ingest
        from docmind.store import CorpusStore
        CONFIG.ensure_dirs();store=CorpusStore()
        sources={'operating-example.txt':'The example operating procedure governs thermal regulation. Coolant temperature must stay below 300 degrees during startup. Operators inspect the coolant pump every shift. The shutdown valve closes within two seconds of the trigger. This is a synthetic demonstration document.',
            'hindi-example.txt':'पुस्तकालय की सामान्य ऋण अवधि 21 दिन है। किताबों को समय पर वापस करना चाहिए। पुस्तकालय सोमवार से शुक्रवार तक खुलता है। यह केवल प्रदर्शन के लिए तैयार किया गया उदाहरण है।'}
        for name,text in sources.items():
            path=CONFIG.data_dir/name;path.write_text(text,encoding='utf-8');ingest(path,store=store)
        out.mkdir(parents=True,exist_ok=True)
        (out/'service.private.json').write_text(json.dumps({'keys':keys,'embedding_model':CONFIG.embedding.model_name,'embedding_revision':CONFIG.embedding.model_revision}))
        environment=dict(os.environ)
        api="from docmind.config import CONFIG; CONFIG.llm.provider='demo'; import uvicorn; uvicorn.run('docmind.api.main:app',host='127.0.0.1',port=8787,proxy_headers=False)"
        processes=[]
        try:
            for label,args in [('api',[sys.executable,'-c',api]),('ui',[sys.executable,'-m','streamlit','run','app.py','--server.address','127.0.0.1','--server.port','8788','--server.headless','true','--browser.gatherUsageStats','false'])]:
                with (out/(label+'.private.log')).open('wb') as log:processes.append(subprocess.Popen(args,cwd=ROOT,env=environment,stdout=log,stderr=log))
            deadline=time.monotonic()+600
            while time.monotonic()<deadline and not (out/'stop.requested').exists():
                if any(process.poll() is not None for process in processes):raise RuntimeError('Demo service exited')
                time.sleep(.2)
        finally:
            for process in processes:
                if process.poll() is None:process.terminate()
            for process in processes:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            (out/'service.private.json').unlink(missing_ok=True)


def capture(out):
    from PIL import Image,ImageDraw,ImageFont
    from playwright.sync_api import sync_playwright
    started=time.monotonic();deadline=started+300
    private=out/'service.private.json'
    while True:
        try:
            credentials=json.loads(private.read_text())
            with urllib.request.urlopen('http://127.0.0.1:8787/health',timeout=2) as response:
                if response.status==200:break
        except (OSError,ValueError):
            if time.monotonic()>deadline:raise TimeoutError('Free demo initialization deadline exceeded')
            time.sleep(.5)
    frames=[];descriptions=[]
    fonts=[Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),Path('/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf')]
    font=next((ImageFont.truetype(str(path),18) for path in fonts if path.exists()),ImageFont.load_default())
    def shot(page,description):
        index=len(frames)+1;raw=out/('ui-'+str(index)+'.png');page.screenshot(path=str(raw))
        with Image.open(raw) as original:
            image=Image.new('RGB',(original.width,original.height+94),'#15243a');image.paste(original,(0,56))
            draw=ImageDraw.Draw(image);draw.text((20,15),description,fill='white',font=font)
            draw.text((20,original.height+65),'Synthetic sources | Real E5 retrieval | Extractive demo reader | No human decisions',fill='white',font=font)
            image.save(out/('frame-'+str(index)+'.png'));frames.append(image.copy());descriptions.append(description)
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch()
        try:
            page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
            page.goto('http://127.0.0.1:8788',wait_until='domcontentloaded',timeout=60000)
            access=page.get_by_label('Personal access key',exact=True);access.fill(credentials['keys']['reader']);access.press('Enter')
            page.get_by_text('Signed in as demo-reader',exact=True).wait_for(timeout=60000)
            assert access.get_attribute('type')=='password'
            shot(page,'1 / 6  Authenticated document library with synthetic English and Hindi sources')
            question=page.get_by_label('Ask a question',exact=True)
            question.fill('What is the coolant temperature limit during startup?')
            shot(page,'2 / 6  Ask a question over the document library')
            page.get_by_role('button',name='Ask',exact=True).click()
            page.get_by_text('Unverified draft. Citation found; automated support has not been established.',exact=True).wait_for(timeout=60000)
            shot(page,'3 / 6  A visible draft is explicitly unverified')
            page.get_by_text('operating-example.txt',exact=True).first.click()
            page.get_by_text('Refresh response',exact=True).first.scroll_into_view_if_needed()
            shot(page,'4 / 6  Inspect the quoted passage and retrieved source')
            question.fill('पुस्तकालय की सामान्य ऋण अवधि कितने दिन है?')
            page.get_by_role('button',name='Ask',exact=True).click()
            page.get_by_text('This response is not released.',exact=True).wait_for(timeout=60000)
            shot(page,'5 / 6  The unvalidated Hindi answer stays held')
            access.fill(credentials['keys']['reviewer']);access.press('Enter')
            page.get_by_text('Signed in as demo-reviewer',exact=True).wait_for(timeout=60000)
            page.get_by_text('Human review',exact=True).scroll_into_view_if_needed()
            page.get_by_text('Choose a decision',exact=True).wait_for(timeout=30000)
            shot(page,'6 / 6  Authorized reviewer view; the item is still pending')
        finally:browser.close()
    frames[0].save(out/'docmind-workflow.gif',save_all=True,append_images=frames[1:],duration=[12500]*6,loop=0,optimize=True,
        comment=b'Offline workflow demonstration; synthetic sources; extractive reader; no hosted inference or human labels.')
    report={'scope':'Actual Streamlit/API workflow, synthetic sources, pinned E5 retrieval and extractive demo generation; not generative QA quality or a production SLA',
        'frame_descriptions':descriptions,'display_duration_ms':75000,'hosted_calls':0,'human_decisions':0,'paid_compute':False,
        'embedding_model':credentials['embedding_model'],'embedding_revision':credentials['embedding_revision'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_commit':os.getenv('GITHUB_SHA'),
        'gif_sha256':hashlib.sha256((out/'docmind-workflow.gif').read_bytes()).hexdigest(),
        'ui_frame_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in out.glob('ui-*.png')},
        'captions_are_not_app_ui':True,
        'frame_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in out.glob('frame-*.png')},
        'capture_elapsed_seconds':time.monotonic()-started}
    (out/'captions.json').write_text(json.dumps({'captions_are_not_app_ui':True,'captions':descriptions},indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');(out/'stop.requested').write_text('Capture complete')
    print(json.dumps({k:report[k] for k in ('display_duration_ms','hosted_calls','human_decisions','paid_compute')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--serve-only',action='store_true');args=parser.parse_args()
    if args.serve_only:serve(args.out.resolve())
    else:capture(args.out.resolve())
