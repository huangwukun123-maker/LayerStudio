"""Loopback workbench, persistent jobs and a single local inference worker."""
from contextlib import asynccontextmanager
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import asyncio
import json
import os
import queue
import re
import threading
import time
import traceback
import uuid
from urllib.parse import urlparse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from engine import Processor
from models import APP_DIR, model_status
from vision import load_image

JOB_ID = re.compile(r'^[a-f0-9]{32}$')
ALLOWED_FILES = {'result.psd','preview.png','original.png','delivery.zip','report.txt'}
MAX_FILE = 20*1024*1024
MAX_BATCH = 80*1024*1024


class JobStore:
    def __init__(self, root, processor):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.processor=processor
        self.lock=threading.RLock()
        self.jobs={}
        self.pending=queue.Queue()
        self.stopping=threading.Event()
        self.worker=None
        self.active_uploads=0
        self.reserved_jobs=0
        for file in self.root.glob('*/status.json'):
            if not JOB_ID.fullmatch(file.parent.name): continue
            try:
                item=json.loads(file.read_text(encoding='utf-8'))
                if item.get('id') != file.parent.name: continue
                if item.get('status') in ('running','queued'):
                    item.update(status='failed',stage='上次运行中断',error='上次处理已中断，请重新上传图片。')
                self.jobs[item['id']]=item
                self._save(item)
            except (OSError,ValueError,KeyError):
                continue

    def _save(self, item):
        directory=self.root/item['id']
        directory.mkdir(exist_ok=True)
        temporary=directory/'status.tmp'
        temporary.write_text(json.dumps(item,ensure_ascii=False),encoding='utf-8')
        temporary.replace(directory/'status.json')

    def start(self):
        self.worker=threading.Thread(target=self._run,name='layerstudio-worker',daemon=True)
        self.worker.start()

    def stop(self):
        self.stopping.set()
        self.pending.put(None)
        if self.worker: self.worker.join(timeout=1)

    def snapshots(self):
        with self.lock:
            return sorted(deepcopy(list(self.jobs.values())),key=lambda j:j['created_at'],reverse=True)

    def snapshot(self, ident):
        if not JOB_ID.fullmatch(ident): raise HTTPException(404,'任务不存在。')
        with self.lock:
            if ident not in self.jobs: raise HTTPException(404,'任务不存在。')
            return deepcopy(self.jobs[ident])

    def add(self, directory, name, options):
        item={'id':directory.name,'name':name,'status':'queued','progress':0,'stage':'等待处理',
              'created_at':datetime.now(timezone.utc).isoformat(),'elapsed':0,'options':options}
        with self.lock:
            self.jobs[item['id']]=item
            self._save(item)
        self.pending.put(item['id'])
        return self.snapshot(item['id'])

    def _run(self):
        while not self.stopping.is_set():
            ident=self.pending.get()
            if ident is None: return
            directory=self.root/ident
            start=time.monotonic()
            def update(progress,stage):
                with self.lock:
                    self.jobs[ident].update(status='running',progress=round(progress),stage=stage,
                                             elapsed=round(time.monotonic()-start,1))
                    self._save(self.jobs[ident])
            try:
                result=self.processor.process(directory/'input',directory,self.jobs[ident]['options'],update)
                result['files']={k:f'/api/jobs/{ident}/files/{v}' for k,v in result['files'].items()}
                with self.lock:
                    self.jobs[ident].update(status='completed',progress=100,stage='处理完成',result=result,
                                             elapsed=round(time.monotonic()-start,1))
                    self._save(self.jobs[ident])
            except Exception as exc:
                (directory/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
                message=str(exc) if isinstance(exc,(ValueError,RuntimeError)) else '处理未完成，请查看任务目录中的error.log或换一张图片重试。'
                with self.lock:
                    self.jobs[ident].update(status='failed',stage='处理失败',error=message,
                                             elapsed=round(time.monotonic()-start,1))
                    self._save(self.jobs[ident])
            finally:
                self.pending.task_done()


def create_app(data_dir=None, processor=None):
    root=Path(data_dir or os.environ.get('LAYERSTUDIO_DATA',APP_DIR/'data'/'jobs'))
    store=JobStore(root,processor if processor is not None else Processor())

    @asynccontextmanager
    async def lifespan(app):
        store.start()
        yield
        store.stop()

    app=FastAPI(title='分层工坊',lifespan=lifespan,docs_url=None,redoc_url=None)
    app.state.store=store
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])

    @app.middleware('http')
    async def local_requests(request,call_next):
        if request.method not in ('GET','HEAD'):
            origin=request.headers.get('origin')
            if origin and urlparse(origin).netloc != request.headers.get('host'):
                return JSONResponse({'detail':'只接受本机软件页面提交的请求。'},status_code=403)
        if request.method not in ('GET','HEAD'):
            # Multipart parsing needs a known length; disallow unbounded chunked requests.
            length=request.headers.get('content-length')
            if request.url.path == '/api/jobs':
                if not length or not length.isdigit():
                    return JSONResponse({'detail':'上传请求缺少有效文件大小。'},status_code=411)
                if int(length)>MAX_BATCH:
                    return JSONResponse({'detail':'一次上传总大小不能超过80MB。'},status_code=413)
        is_upload=request.method=='POST' and request.url.path=='/api/jobs'
        if is_upload:
            with store.lock:
                if store.stopping.is_set():
                    return JSONResponse({'detail':'软件正在退出，请重新启动后上传。'},status_code=503)
                store.active_uploads+=1
        try:
            response=await call_next(request)
        finally:
            if is_upload:
                with store.lock: store.active_uploads-=1
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Cache-Control']='no-store'
        return response

    @app.get('/api/health')
    def health():
        status=model_status()
        models_ready=all(x['ready'] for x in status)
        missing='、'.join(x['name'] for x in status if not x['ready'])
        return {'ready':models_ready,'name':'分层工坊','version':'0.3.5-oss.1',
                'edition':'opensource','models':status,
                'message':'本机模型就绪，图片无需上传云端。' if models_ready else f'缺少模型：{missing}。请参阅 docs/MODELS.md。'}

    @app.post('/api/shutdown')
    def shutdown():
        with store.lock:
            if store.active_uploads or any(j['status'] in ('queued','running') for j in store.snapshots()):
                raise HTTPException(409,'还有图片正在上传或处理，请等任务完成后再退出。')
            callback=getattr(app.state,'shutdown_callback',None)
            if callback is None:
                raise HTTPException(409,'当前以开发模式运行，请关闭启动窗口来退出。')
            store.stopping.set()
        threading.Timer(.5,callback).start()
        return {'message':'软件已退出，下次双击启动程序即可打开。'}

    @app.get('/api/jobs')
    def jobs(): return {'jobs':store.snapshots()}

    @app.get('/api/jobs/{ident}')
    def job(ident:str): return store.snapshot(ident)

    @app.post('/api/jobs')
    async def upload(files:list[UploadFile]=File(...),scale:int=Form(3),crop_ui:bool=Form(True),split_light:bool=Form(True)):
        if scale not in (2,3,4): raise HTTPException(400,'请选择2、3或4倍清晰化。')
        if not 1<=len(files)<=10: raise HTTPException(400,'每次请选择1–10张图片。')
        with store.lock:
            pending=sum(j['status'] in ('running','queued') for j in store.jobs.values())
            if pending+store.reserved_jobs+len(files)>30:
                raise HTTPException(429,'已有较多图片等待处理，请稍后再上传。')
            store.reserved_jobs+=len(files)
        try:
            prepared=[]
            try:
                for upload in files:
                    directory=store.root/uuid.uuid4().hex
                    directory.mkdir()
                    prepared.append((directory,upload))
                    count=0
                    with (directory/'input').open('wb') as dest:
                        while block:=await upload.read(1024*1024):
                            count+=len(block)
                            if count>MAX_FILE: raise HTTPException(413,'单张图片不能超过20MB。')
                            dest.write(block)
                    await asyncio.to_thread(load_image,directory/'input')
            except (ValueError,HTTPException) as exc:
                # Only our newly allocated UUID directories contain these temporary files.
                for directory,_ in prepared:
                    for file in directory.iterdir(): file.unlink()
                    directory.rmdir()
                if isinstance(exc,HTTPException): raise
                raise HTTPException(400,str(exc)) from exc
            finally:
                for file in files: await file.close()
            result=[]
            for directory,upload in prepared:
                raw=(upload.filename or '图片').replace('\\','/').split('/')[-1]
                name=re.sub(r'[\x00-\x1f<>:"|?*]','_',raw)[:100] or '图片'
                result.append(store.add(directory,name,{'scale':scale,'crop_ui':crop_ui,'split_light':split_light}))
            return {'jobs':result}
        finally:
            with store.lock: store.reserved_jobs-=len(files)

    @app.get('/api/jobs/{ident}/files/{filename}')
    def artifact(ident:str,filename:str):
        item=store.snapshot(ident)
        if item['status']!='completed' or filename not in ALLOWED_FILES:
            raise HTTPException(404,'文件不存在或尚未生成。')
        path=store.root/ident/filename
        if not path.is_file(): raise HTTPException(404,'文件已被移动或删除。')
        names={'result.psd':'可编辑分层.psd','preview.png':'合成预览.png','original.png':'输入原图.png',
               'delivery.zip':'PSD交付包.zip','report.txt':'处理说明.txt'}
        return FileResponse(path,filename=names[filename],content_disposition_type='inline' if filename.endswith('.png') else 'attachment')

    static=APP_DIR/'static'
    if static.is_dir():
        app.mount('/static',StaticFiles(directory=static),name='static')
        @app.get('/')
        def index(): return FileResponse(static/'index.html')
        @app.get('/history')
        def history_page(): return FileResponse(static/'index.html')
    return app


if __name__=='__main__':
    import uvicorn
    uvicorn.run(create_app(),host='127.0.0.1',port=int(os.environ.get('LAYERSTUDIO_PORT','8765')))
