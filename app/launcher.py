"""Double-click launcher; no terminal, assistant or API account is required."""
import json
import logging
import os
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from models import APP_DIR


def main():
    logdir=APP_DIR/'data'
    logdir.mkdir(exist_ok=True)
    # Windowed executables have no stdout/stderr. Libraries still log there.
    stream=open(logdir/'software.log','a',encoding='utf-8',buffering=1)
    sys.stdout=stream
    sys.stderr=stream
    logging.basicConfig(stream=stream,level=logging.INFO)
    requested=int(os.environ.get('LAYERSTUDIO_PORT','8765'))
    selected=None
    for port in range(requested,requested+10):
        url=f'http://127.0.0.1:{port}'
        try:
            with urllib.request.urlopen(url+'/api/health',timeout=.35) as reply:
                existing=json.load(reply)
                if existing.get('name')=='分层工坊' and existing.get('edition')=='opensource':
                    if os.environ.get('LAYERSTUDIO_NO_BROWSER')!='1': webbrowser.open(url)
                    return
        except Exception:
            pass
    for port in range(requested,requested+10):
        with socket.socket() as probe:
            try: probe.bind(('127.0.0.1',port))
            except OSError: continue
        selected=port
        break
    if selected is None:
        raise RuntimeError('本机端口被占用，请稍后再启动。')
    from server import create_app
    import uvicorn
    app=create_app()
    config=uvicorn.Config(app,host='127.0.0.1',port=selected,log_config=None,access_log=False)
    server=uvicorn.Server(config)
    app.state.shutdown_callback=lambda:setattr(server,'should_exit',True)
    def open_when_ready():
        for _ in range(120):
            if server.started:
                if os.environ.get('LAYERSTUDIO_NO_BROWSER')!='1':
                    webbrowser.open(f'http://127.0.0.1:{selected}')
                return
            time.sleep(.25)
    threading.Thread(target=open_when_ready,daemon=True).start()
    server.run()


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if os.name=='nt':
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,'启动失败：'+str(exc)+'\n请查看data/software.log。','分层工坊',16)
