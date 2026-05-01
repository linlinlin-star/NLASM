import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NLASM Registry", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = Path(os.environ.get("NLASM_REGISTRY_DIR", "./registry-data"))
AUTH_TOKENS: set[str] = set(os.environ.get("NLASM_REGISTRY_TOKENS", "").split(",")) - {""}


def _pkg_dir(name: str) -> Path:
    return STORAGE_DIR / "packages" / name


def _ver_dir(name: str, version: str) -> Path:
    return _pkg_dir(name) / "versions" / version


def _check_auth(request: Request) -> None:
    if not AUTH_TOKENS:
        return
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = auth
    if token not in AUTH_TOKENS:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _compute_hash(data: bytes) -> str:
    return "sha256-" + hashlib.sha256(data).hexdigest()


@app.get("/api/packages")
async def search_packages(q: str = "", limit: int = 20, offset: int = 0):
    packages_dir = STORAGE_DIR / "packages"
    if not packages_dir.exists():
        return {"results": [], "total": 0}
    results = []
    for pkg_path in sorted(packages_dir.iterdir()):
        if not pkg_path.is_dir():
            continue
        meta_file = pkg_path / "meta.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if q and q.lower() not in meta.get("name", "").lower() and q.lower() not in meta.get("description", "").lower():
            continue
        results.append(meta)
    total = len(results)
    results = results[offset: offset + limit]
    return {"results": results, "total": total}


@app.get("/api/packages/{name}")
async def get_package_info(name: str):
    pkg = _pkg_dir(name)
    meta_file = pkg / "meta.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail=f"Package '{name}' not found")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    versions_dir = pkg / "versions"
    versions = []
    if versions_dir.exists():
        for v in sorted(versions_dir.iterdir()):
            if v.is_dir():
                vm = v / "manifest.json"
                if vm.exists():
                    versions.append(json.loads(vm.read_text(encoding="utf-8")))
    meta["versions"] = versions
    return meta


@app.get("/api/packages/{name}/{version}")
async def get_package_version(name: str, version: str):
    ver = _ver_dir(name, version)
    mf = ver / "manifest.json"
    if not mf.exists():
        raise HTTPException(status_code=404, detail=f"Version '{version}' of '{name}' not found")
    return json.loads(mf.read_text(encoding="utf-8"))


@app.get("/api/packages/{name}/{version}/download")
async def download_package(name: str, version: str):
    ver = _ver_dir(name, version)
    zip_file = ver / f"{name}-{version}.zip"
    if not zip_file.exists():
        raise HTTPException(status_code=404, detail=f"Package archive not found")
    return FileResponse(
        path=str(zip_file),
        filename=f"{name}-{version}.zip",
        media_type="application/zip",
    )


@app.post("/api/packages/{name}")
async def publish_package(name: str, request: Request, file: UploadFile = File(...)):
    _check_auth(request)
    data = await file.read()
    integrity = _compute_hash(data)

    tmp_dir = STORAGE_DIR / "tmp" / uuid.uuid4().hex
    tmp_dir.mkdir(parents=True, exist_ok=True)

    zip_path = tmp_dir / "package.zip"
    zip_path.write_bytes(data)

    import zipfile
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(tmp_dir / "contents"))
    except zipfile.BadZipFile:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Invalid zip file")

    manifest_file = None
    for f in (tmp_dir / "contents").rglob("nlasm.json"):
        manifest_file = f
        break

    if not manifest_file:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Missing nlasm.json manifest")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    pkg_name = manifest.get("name", name)
    version = manifest.get("version", "0.0.0")

    ver_dir = _ver_dir(pkg_name, version)
    ver_dir.mkdir(parents=True, exist_ok=True)

    dest_zip = ver_dir / f"{pkg_name}-{version}.zip"
    shutil.copy2(str(zip_path), str(dest_zip))

    for f in (tmp_dir / "contents").rglob("*"):
        if f.is_file() and f.name != "nlasm.json":
            rel = f.relative_to(tmp_dir / "contents")
            dest = ver_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(f), str(dest))

    (ver_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    pkg = _pkg_dir(pkg_name)
    meta_file = pkg / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    else:
        meta = {
            "name": pkg_name,
            "description": manifest.get("description", ""),
            "author": manifest.get("author", ""),
            "latest": version,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    meta["latest"] = version
    meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"status": "ok", "name": pkg_name, "version": version, "integrity": integrity}


@app.delete("/api/packages/{name}/{version}")
async def unpublish_package(name: str, version: str, request: Request):
    _check_auth(request)
    ver = _ver_dir(name, version)
    if not ver.exists():
        raise HTTPException(status_code=404, detail="Version not found")
    shutil.rmtree(ver)
    return {"status": "ok", "deleted": f"{name}@{version}"}


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("NLASM_REGISTRY_PORT", "8420"))
    uvicorn.run(app, host="0.0.0.0", port=port)
