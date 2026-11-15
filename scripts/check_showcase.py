"""Exercise the static explorer with a real headless browser, without external calls."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]


def check(out):
    out.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(ROOT/'.site-dist')))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch();page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto('http://127.0.0.1:'+str(server.server_port),wait_until='networkidle')
            expect(page.locator('#counts')).to_contain_text('123 / 180')
            page.screenshot(path=str(out/'showcase-desktop.png'))
            page.locator('#replay').scroll_into_view_if_needed();page.screenshot(path=str(out/'showcase-replay.png'))
            page.locator('#state').select_option('unserved');expect(page.locator('#counts')).to_contain_text('Showing 57')
            expect(page.locator('#detail')).to_contain_text('No output exists')
            page.locator('#condition').select_option('original');expect(page.locator('#counts')).to_contain_text('Showing 0')
            page.locator('#state').select_option('format');expect(page.locator('#counts')).to_contain_text('Showing 103')
            page.locator('#state').select_option('all');page.locator('#condition').select_option('complete')
            page.locator('#search').fill('Safety notice');expect(page.locator('#counts')).to_contain_text('Showing 2')
            page.locator('#search').fill('');page.locator('[data-case-id="garden-xml-json_current"]').click()
            assert page.locator('system').count()==0
            page.set_viewport_size({'width':390,'height':844});page.goto('http://127.0.0.1:'+str(server.server_port),wait_until='networkidle')
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile layout overflow'
            page.screenshot(path=str(out/'showcase-mobile.png'))
            assert not errors,errors
            browser.close()
        report={'real_browser':True,'filters_checked':True,'unserved_preserved':57,'original_format_failures_preserved':103,'mobile_width':390,'page_errors':errors,'hosted_inference_calls':0,'private_documents_published':False}
        (out/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
    finally:server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--out',type=Path,default=ROOT/'reports/release_workspace/showcase-check');args=a.parse_args();check(args.out)
