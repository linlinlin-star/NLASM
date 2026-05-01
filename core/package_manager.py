from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


MANIFEST_FILE = "nlasm.json"
LOCK_FILE = "nlasm.lock"
PACKAGES_DIR = ".nlasm"
GLOBAL_CACHE_DIR = Path.home() / ".nlasm" / "cache"
REGISTRY_CONFIG_DIR = Path.home() / ".nlasm"
REGISTRY_CONFIG_FILE = REGISTRY_CONFIG_DIR / "registries.json"
DEFAULT_REGISTRY = "https://registry.nlasm.dev"


class SemVer:
    __slots__ = ("major", "minor", "patch", "prerelease", "build")

    def __init__(self, major: int, minor: int, patch: int,
                 prerelease: str = "", build: str = ""):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.prerelease = prerelease
        self.build = build

    @classmethod
    def parse(cls, version: str) -> SemVer:
        s = version.strip().lstrip("v")
        m = re.match(
            r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?$",
            s,
        )
        if not m:
            raise ValueError(f"无效的语义版本: {version}")
        return cls(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            m.group(4) or "", m.group(5) or "",
        )

    def __str__(self) -> str:
        s = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            s += f"-{self.prerelease}"
        if self.build:
            s += f"+{self.build}"
        return s

    def __repr__(self) -> str:
        return f"SemVer({self})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented
        return (self.major, self.minor, self.patch, self.prerelease) == \
               (other.major, other.minor, other.patch, other.prerelease)

    def __lt__(self, other: SemVer) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self._compare_prerelease(self.prerelease, other.prerelease) < 0
        return False

    @staticmethod
    def _prerelease_priority(tag: str) -> int:
        t = tag.lower()
        if t in ("dev", "d"):
            return 0
        if t.startswith("alpha") or t.startswith("a"):
            return 1
        if t.startswith("beta") or t.startswith("b"):
            return 2
        if t.startswith("rc") or t.startswith("c"):
            return 3
        if t.startswith("preview"):
            return 2
        if t.startswith("pre"):
            return 2
        return -1

    @staticmethod
    def _compare_prerelease(a: str, b: str) -> int:
        pa = a.split(".")
        pb = b.split(".")
        for i in range(max(len(pa), len(pb))):
            sa = pa[i] if i < len(pa) else ""
            sb = pb[i] if i < len(pb) else ""
            na = SemVer._prerelease_priority(sa)
            nb = SemVer._prerelease_priority(sb)
            if na >= 0 and nb >= 0:
                if na != nb:
                    return na - nb
                continue
            if na >= 0:
                return -1
            if nb >= 0:
                return 1
            try:
                ia = int(sa)
                ib = int(sb)
                if ia != ib:
                    return ia - ib
            except ValueError:
                if sa < sb:
                    return -1
                if sa > sb:
                    return 1
        return 0

    def __le__(self, other: SemVer) -> bool:
        return self == other or self < other

    def __gt__(self, other: SemVer) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented
        return other < self

    def __ge__(self, other: SemVer) -> bool:
        return self == other or self > other

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))


class VersionRange:
    def __init__(self, spec: str):
        self.spec = spec.strip()
        self._matchers = self._parse(self.spec)

    def _parse(self, spec: str) -> list:
        matchers: list = []
        for part in spec.split("||"):
            part = part.strip()
            if not part:
                continue
            matchers.append(self._parse_single(part))
        return matchers

    def _parse_single(self, part: str) -> list:
        constraints: list = []
        for token in part.split():
            token = token.strip()
            if not token:
                continue
            if token.startswith("^"):
                v = SemVer.parse(token[1:])
                constraints.append(("caret", v))
            elif token.startswith("~"):
                v = SemVer.parse(token[1:])
                constraints.append(("tilde", v))
            elif token.startswith(">="):
                v = SemVer.parse(token[2:])
                constraints.append(("gte", v))
            elif token.startswith("<="):
                v = SemVer.parse(token[2:])
                constraints.append(("lte", v))
            elif token.startswith(">"):
                v = SemVer.parse(token[1:])
                constraints.append(("gt", v))
            elif token.startswith("<"):
                v = SemVer.parse(token[1:])
                constraints.append(("lt", v))
            elif token == "*":
                constraints.append(("any", None))
            else:
                v = SemVer.parse(token)
                constraints.append(("exact", v))
        return constraints

    def matches(self, version: SemVer) -> bool:
        if not self._matchers:
            return True
        for group in self._matchers:
            if self._group_matches(group, version):
                return True
        return False

    def _group_matches(self, group: list, version: SemVer) -> bool:
        for kind, v in group:
            if kind == "any":
                if version.prerelease:
                    return False
                continue
            elif kind == "exact":
                if version != v:
                    return False
            elif kind == "gt":
                if not version > v:
                    return False
            elif kind == "lt":
                if not version < v:
                    return False
            elif kind == "gte":
                if not version >= v:
                    return False
            elif kind == "lte":
                if not version <= v:
                    return False
            elif kind == "caret":
                if version < v:
                    return False
                if version.prerelease and not v.prerelease:
                    if (version.major, version.minor, version.patch) != (v.major, v.minor, v.patch):
                        return False
                if v.major != 0:
                    if version.major != v.major:
                        return False
                elif v.minor != 0:
                    if version.major != v.major or version.minor != v.minor:
                        return False
                else:
                    if version != v:
                        return False
            elif kind == "tilde":
                if version < v:
                    return False
                if version.prerelease and not v.prerelease:
                    if (version.major, version.minor, version.patch) != (v.major, v.minor, v.patch):
                        return False
                if version.major != v.major or version.minor != v.minor:
                    return False
        return True

    def __repr__(self) -> str:
        return f"VersionRange({self.spec!r})"


@dataclass
class PackageManifest:
    name: str = ""
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    license: str = ""
    main: str = ""
    dependencies: dict[str, str] = field(default_factory=dict)
    dev_dependencies: dict[str, str] = field(default_factory=dict)
    nlasm_version: str = ""
    keywords: list[str] = field(default_factory=list)
    repository: str = ""
    homepage: str = ""
    registry: str = ""
    git: str = ""
    scripts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "version": self.version}
        if self.description:
            d["description"] = self.description
        if self.author:
            d["author"] = self.author
        if self.license:
            d["license"] = self.license
        if self.main:
            d["main"] = self.main
        if self.dependencies:
            d["dependencies"] = dict(sorted(self.dependencies.items()))
        if self.dev_dependencies:
            d["devDependencies"] = dict(sorted(self.dev_dependencies.items()))
        if self.nlasm_version:
            d["nlasmVersion"] = self.nlasm_version
        if self.keywords:
            d["keywords"] = self.keywords
        if self.repository:
            d["repository"] = self.repository
        if self.homepage:
            d["homepage"] = self.homepage
        if self.registry:
            d["registry"] = self.registry
        if self.git:
            d["git"] = self.git
        if self.scripts:
            d["scripts"] = dict(sorted(self.scripts.items()))
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PackageManifest:
        return cls(
            name=d.get("name", ""),
            version=d.get("version", "0.1.0"),
            description=d.get("description", ""),
            author=d.get("author", ""),
            license=d.get("license", ""),
            main=d.get("main", ""),
            dependencies=d.get("dependencies", {}),
            dev_dependencies=d.get("devDependencies", {}),
            nlasm_version=d.get("nlasmVersion", ""),
            keywords=d.get("keywords", []),
            repository=d.get("repository", ""),
            homepage=d.get("homepage", ""),
            registry=d.get("registry", ""),
            git=d.get("git", ""),
            scripts=d.get("scripts", {}),
        )

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> PackageManifest:
        if not path.exists():
            raise FileNotFoundError(f"找不到包配置文件: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class LockEntry:
    name: str
    version: str
    resolved: str
    integrity: str = ""
    dependencies: dict[str, str] = field(default_factory=dict)
    source: str = "registry"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "resolved": self.resolved,
        }
        if self.integrity:
            d["integrity"] = self.integrity
        if self.dependencies:
            d["dependencies"] = dict(sorted(self.dependencies.items()))
        if self.source != "registry":
            d["source"] = self.source
        return d

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> LockEntry:
        return cls(
            name=name,
            version=d.get("version", ""),
            resolved=d.get("resolved", ""),
            integrity=d.get("integrity", ""),
            dependencies=d.get("dependencies", {}),
            source=d.get("source", "registry"),
        )


@dataclass
class LockFile:
    version: int = 1
    packages: dict[str, LockEntry] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        pkgs: dict[str, Any] = {}
        for name in sorted(self.packages):
            pkgs[name] = self.packages[name].to_dict()
        return {"version": self.version, "packages": pkgs}

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> LockFile:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        lf = cls(version=data.get("version", 1))
        for name, entry in data.get("packages", {}).items():
            lf.packages[name] = LockEntry.from_dict(name, entry)
        return lf


@dataclass
class RegistryConfig:
    name: str
    url: str
    mirror_of: str = ""
    priority: int = 0
    auth_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "url": self.url}
        if self.mirror_of:
            d["mirrorOf"] = self.mirror_of
        if self.priority:
            d["priority"] = self.priority
        if self.auth_token:
            d["authToken"] = self.auth_token
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegistryConfig:
        return cls(
            name=d.get("name", ""),
            url=d.get("url", ""),
            mirror_of=d.get("mirrorOf", ""),
            priority=d.get("priority", 0),
            auth_token=d.get("authToken", ""),
        )


class RegistryManager:
    def __init__(self, local_registry: LocalRegistry | None = None) -> None:
        self._registries: list[RegistryConfig] = []
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl: dict[str, float] = {}
        self._cache_duration = 300.0
        self._local_registry = local_registry or LocalRegistry()
        self._load_config()

    def _load_config(self) -> None:
        REGISTRY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if REGISTRY_CONFIG_FILE.exists():
            try:
                data = json.loads(REGISTRY_CONFIG_FILE.read_text(encoding="utf-8"))
                for item in data.get("registries", []):
                    self._registries.append(RegistryConfig.from_dict(item))
            except (json.JSONDecodeError, OSError):
                pass
        if not self._registries:
            self._registries.append(
                RegistryConfig(name="official", url=DEFAULT_REGISTRY, priority=0)
            )

    def _save_config(self) -> None:
        REGISTRY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {"registries": [r.to_dict() for r in self._registries]}
        REGISTRY_CONFIG_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def add_registry(self, name: str, url: str, mirror_of: str = "",
                     priority: int = 0, auth_token: str = "") -> None:
        for r in self._registries:
            if r.name == name:
                r.url = url
                r.mirror_of = mirror_of
                r.priority = priority
                r.auth_token = auth_token
                self._save_config()
                return
        self._registries.append(
            RegistryConfig(name=name, url=url, mirror_of=mirror_of,
                           priority=priority, auth_token=auth_token)
        )
        self._save_config()

    def remove_registry(self, name: str) -> bool:
        before = len(self._registries)
        self._registries = [r for r in self._registries if r.name != name]
        if len(self._registries) < before:
            self._save_config()
            return True
        return False

    def list_registries(self) -> list[RegistryConfig]:
        return sorted(self._registries, key=lambda r: r.priority)

    def get_primary(self) -> RegistryConfig:
        return min(self._registries, key=lambda r: r.priority) if self._registries else \
            RegistryConfig(name="official", url=DEFAULT_REGISTRY)

    def _is_cache_valid(self, cache_key: str) -> bool:
        import time
        if cache_key not in self._cache_ttl:
            return False
        return time.time() - self._cache_ttl[cache_key] < self._cache_duration

    def _set_cache(self, cache_key: str, data: dict[str, Any]) -> None:
        import time
        self._cache[cache_key] = data
        self._cache_ttl[cache_key] = time.time()

    def _get_cache(self, cache_key: str) -> dict[str, Any] | None:
        if self._is_cache_valid(cache_key):
            return self._cache.get(cache_key)
        return None

    def invalidate_cache(self, pattern: str = "") -> None:
        if not pattern:
            self._cache.clear()
            self._cache_ttl.clear()
        else:
            keys_to_remove = [k for k in self._cache if pattern in k]
            for k in keys_to_remove:
                del self._cache[k]
                self._cache_ttl.pop(k, None)

    def fetch_package_info(self, name: str, prefer_local: bool = True) -> dict[str, Any]:
        cache_key = f"info:{name}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        errors: list[str] = []

        if prefer_local:
            try:
                info = self._local_registry.get_package_info(name)
                self._set_cache(cache_key, info)
                return info
            except FileNotFoundError:
                pass

        for reg in sorted(self._registries, key=lambda r: r.priority):
            try:
                registry = PackageRegistry(reg.url, auth_token=reg.auth_token)
                info = registry.get_package_info(name)
                self._set_cache(cache_key, info)
                return info
            except (URLError, HTTPError) as e:
                errors.append(f"{reg.name}({reg.url}): {e}")
                continue

        if errors:
            raise RuntimeError(
                f"所有注册表均无法获取包 {name} 的信息:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
        raise FileNotFoundError(f"找不到包: {name}")

    def resolve_version(self, name: str, version_range: str,
                        prefer_local: bool = True) -> tuple[str, str]:
        range_ = VersionRange(version_range)

        if prefer_local:
            try:
                versions = self._local_registry.get_versions(name)
                parsed = sorted([SemVer.parse(v) for v in versions], reverse=True)
                for v in parsed:
                    if range_.matches(v):
                        return str(v), f"local:{self._local_registry.base_dir}"
            except (FileNotFoundError, ValueError):
                pass

        for reg in sorted(self._registries, key=lambda r: r.priority):
            try:
                registry = PackageRegistry(reg.url, auth_token=reg.auth_token)
                version = registry.resolve_version(name, version_range)
                return version, reg.url
            except (ValueError, URLError, HTTPError):
                continue

        raise ValueError(f"无法解析包 {name}@{version_range}")


class PackageRegistry:
    def __init__(self, registry_url: str = DEFAULT_REGISTRY, timeout: int = 30,
                 auth_token: str = ""):
        self.registry_url = registry_url.rstrip("/")
        self.timeout = timeout
        self.auth_token = auth_token

    def _fetch_json(self, url: str) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        req = Request(url, headers=headers)
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_package_info(self, name: str) -> dict[str, Any]:
        url = f"{self.registry_url}/api/packages/{name}"
        return self._fetch_json(url)

    def get_versions(self, name: str) -> list[str]:
        info = self.get_package_info(name)
        versions_data = info.get("versions", [])
        if isinstance(versions_data, list):
            return [v.get("version", v.get("name", "")) if isinstance(v, dict) else str(v) for v in versions_data]
        return list(versions_data.keys())

    def resolve_version(self, name: str, version_range: str) -> str:
        versions = self.get_versions(name)
        if not versions:
            raise ValueError(f"包 {name} 没有可用版本")
        range_ = VersionRange(version_range)
        parsed = sorted(
            [SemVer.parse(v) for v in versions],
            reverse=True,
        )
        for v in parsed:
            if range_.matches(v):
                return str(v)
        raise ValueError(
            f"包 {name} 没有匹配 {version_range} 的版本 (可用: {', '.join(versions)})"
        )

    def download_package(self, name: str, version: str, dest: Path) -> Path:
        url = f"{self.registry_url}/api/packages/{name}/{version}/download"
        headers: dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        req = Request(url, headers=headers)
        with urlopen(req, timeout=self.timeout) as resp:
            data = resp.read()
        tarball = dest / f"{name}-{version}.zip"
        tarball.write_bytes(data)
        return tarball

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        url = f"{self.registry_url}/api/packages?q={query}&limit={limit}"
        try:
            result = self._fetch_json(url)
            return result.get("results", [])
        except Exception:
            return []

    def publish(self, package_dir: Path, manifest: PackageManifest) -> bool:
        import zipfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="nlasm-publish-"))
        try:
            zip_path = tmp_dir / f"{manifest.name}-{manifest.version}.zip"
            with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
                for f in package_dir.rglob("*"):
                    if f.is_file() and ".git" not in str(f) and f.name != "nlasm.lock":
                        arcname = str(f.relative_to(package_dir))
                        zf.write(str(f), arcname)
            url = f"{self.registry_url}/api/packages/{manifest.name}"
            headers: dict[str, str] = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            data = zip_path.read_bytes()
            boundary = uuid.uuid4().hex
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{manifest.name}-{manifest.version}.zip"\r\n'
                f"Content-Type: application/zip\r\n\r\n"
            ).encode()
            body += data
            body += f"\r\n--{boundary}--\r\n".encode()
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
            headers["Content-Length"] = str(len(body))
            req = Request(url, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                return resp_data.get("status") == "ok"
        except Exception as e:
            print(f"  [发布] 上传失败: {e}")
            return False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class LocalRegistry:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (Path.home() / ".nlasm" / "local-registry")

    def get_package_info(self, name: str) -> dict[str, Any]:
        pkg_dir = self.base_dir / name
        manifest_path = pkg_dir / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"本地包不存在: {name}")
        manifest = PackageManifest.load(manifest_path)
        versions_dir = pkg_dir / "versions"
        versions: dict[str, Any] = {}
        if versions_dir.exists():
            for vdir in sorted(versions_dir.iterdir()):
                vm_path = vdir / MANIFEST_FILE
                if vm_path.exists():
                    vm = PackageManifest.load(vm_path)
                    versions[vm.version] = {"manifest": vm.to_dict()}
        return {
            "name": manifest.name,
            "description": manifest.description,
            "versions": versions,
        }

    def get_versions(self, name: str) -> list[str]:
        info = self.get_package_info(name)
        return list(info.get("versions", {}).keys())

    def resolve_version(self, name: str, version_range: str) -> str:
        versions = self.get_versions(name)
        if not versions:
            raise ValueError(f"本地包 {name} 没有可用版本")
        range_ = VersionRange(version_range)
        parsed = sorted([SemVer.parse(v) for v in versions], reverse=True)
        for v in parsed:
            if range_.matches(v):
                return str(v)
        raise ValueError(
            f"本地包 {name} 没有匹配 {version_range} 的版本"
        )

    def download_package(self, name: str, version: str, dest: Path) -> Path:
        src = self.base_dir / name / "versions" / version
        if not src.exists():
            raise FileNotFoundError(f"本地包 {name}@{version} 不存在")
        tarball = dest / f"{name}-{version}.zip"
        with zipfile.ZipFile(tarball, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(src))
        return tarball

    def publish(self, package_dir: Path, manifest: PackageManifest) -> None:
        name = manifest.name
        version = manifest.version
        dest_dir = self.base_dir / name / "versions" / version
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in package_dir.rglob("*.nl"):
            if f.name == MANIFEST_FILE:
                continue
            rel = f.relative_to(package_dir)
            target = dest_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
        manifest.save(dest_dir / MANIFEST_FILE)
        top_manifest = self.base_dir / name / MANIFEST_FILE
        manifest.save(top_manifest)


@dataclass
class DependencyNode:
    name: str
    version_range: str
    resolved_version: str = ""
    source: str = "registry"
    dependencies: dict[str, str] = field(default_factory=dict)
    depth: int = 0
    parent: str = ""


@dataclass
class ConflictInfo:
    package: str
    requested_by: list[tuple[str, str]]
    versions: list[str]
    message: str = ""


class VersionResolver:
    def __init__(self, registry_manager: RegistryManager,
                 local_registry: LocalRegistry | None = None):
        self.registry_manager = registry_manager
        self.local_registry = local_registry or LocalRegistry()
        self._cache: dict[str, str] = {}
        self._conflicts: list[ConflictInfo] = []
        self._resolution_tree: dict[str, DependencyNode] = {}

    @property
    def conflicts(self) -> list[ConflictInfo]:
        return list(self._conflicts)

    def clear(self) -> None:
        self._cache.clear()
        self._conflicts.clear()
        self._resolution_tree.clear()

    def resolve(self, dependencies: dict[str, str]) -> dict[str, str]:
        self._conflicts.clear()
        self._resolution_tree.clear()

        resolved: dict[str, str] = {}
        version_requests: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for name, vr in dependencies.items():
            version_requests[name].append(("(root)", vr))

        queue: list[tuple[str, str, int, str]] = [
            (name, vr, 0, "(root)") for name, vr in dependencies.items()
        ]
        visited: set[str] = set()
        in_progress: set[str] = set()

        while queue:
            name, version_range, depth, parent = queue.pop(0)

            if name in resolved:
                existing = SemVer.parse(resolved[name])
                new_range = VersionRange(version_range)
                if new_range.matches(existing):
                    version_requests[name].append((parent, version_range))
                    continue
                else:
                    self._handle_conflict(name, version_requests[name], resolved[name], version_range)
                    continue

            if name in in_progress:
                continue

            in_progress.add(name)
            version_requests[name].append((parent, version_range))

            cache_key = f"{name}@{version_range}"
            if cache_key in self._cache:
                resolved[name] = self._cache[cache_key]
                self._resolution_tree[name] = DependencyNode(
                    name=name, version_range=version_range,
                    resolved_version=resolved[name], depth=depth, parent=parent,
                )
                in_progress.discard(name)
                visited.add(name)
                continue

            try:
                version, source = self.registry_manager.resolve_version(
                    name, version_range
                )
            except (ValueError, FileNotFoundError, RuntimeError, URLError, HTTPError) as e:
                print(f"[警告] 无法解析包 {name}@{version_range}: {e}")
                in_progress.discard(name)
                continue

            resolved[name] = version
            self._cache[cache_key] = version
            self._resolution_tree[name] = DependencyNode(
                name=name, version_range=version_range,
                resolved_version=version, source=source,
                depth=depth, parent=parent,
            )

            try:
                info = self.registry_manager.fetch_package_info(name)
                version_info = info.get("versions", {}).get(version, {})
                manifest_data = version_info.get("manifest", {})
                sub_deps = manifest_data.get("dependencies", {})
                self._resolution_tree[name].dependencies = dict(sub_deps)

                for dep_name, dep_range in sub_deps.items():
                    if dep_name not in visited and dep_name not in in_progress:
                        queue.append((dep_name, dep_range, depth + 1, name))
            except (FileNotFoundError, RuntimeError, URLError, HTTPError):
                pass

            in_progress.discard(name)
            visited.add(name)

        return resolved

    def _handle_conflict(self, name: str, requests: list[tuple[str, str]],
                         current_version: str, new_range: str) -> None:
        existing_range = requests[0][1] if requests else "*"
        try:
            all_versions_info = self.registry_manager.fetch_package_info(name)
            all_versions = list(all_versions_info.get("versions", {}).keys())
        except (FileNotFoundError, RuntimeError, URLError, HTTPError):
            all_versions = [current_version]

        conflict = ConflictInfo(
            package=name,
            requested_by=requests + [("(conflict)", new_range)],
            versions=all_versions,
            message=(
                f"依赖冲突: {name} 已解析为 {current_version} "
                f"(匹配 {existing_range})，但 {new_range} 需要不同版本"
            ),
        )
        self._conflicts.append(conflict)
        print(f"[冲突] {conflict.message}")

    def get_resolution_tree(self) -> dict[str, DependencyNode]:
        return dict(self._resolution_tree)

    def print_dependency_tree(self, dependencies: dict[str, str],
                              resolved: dict[str, str] | None = None) -> str:
        if resolved is None:
            resolved = {}
        lines: list[str] = []
        self._build_tree_lines(dependencies, resolved, lines, "", 0)
        return "\n".join(lines)

    def _build_tree_lines(
        self,
        deps: dict[str, str],
        resolved: dict[str, str],
        lines: list[str],
        prefix: str,
        depth: int,
    ) -> None:
        items = sorted(deps.items())
        for i, (name, vr) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            resolved_ver = resolved.get(name, "?")
            lines.append(f"{prefix}{connector}{name}@{vr} -> {resolved_ver}")

            node = self._resolution_tree.get(name)
            if node and node.dependencies:
                child_prefix = prefix + ("    " if is_last else "│   ")
                self._build_tree_lines(node.dependencies, resolved, lines, child_prefix, depth + 1)


def compute_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return f"sha256-{h.hexdigest()}"


def verify_integrity(path: Path, expected: str) -> bool:
    if not expected:
        return True
    actual = compute_hash(path)
    return actual == expected


def extract_zip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def is_git_url(spec: str) -> bool:
    return spec.startswith("git+") or spec.startswith("git://") or \
           spec.endswith(".git") or "github.com" in spec or "gitlab.com" in spec


def parse_git_spec(spec: str) -> tuple[str, str]:
    url = spec
    version = ""
    if spec.startswith("git+"):
        url = spec[4:]
    if "#" in url:
        url, version = url.rsplit("#", 1)
    return url, version


class GitInstaller:
    @staticmethod
    def install(url: str, version: str, dest: Path) -> bool:
        try:
            cmd = ["git", "clone", "--depth", "1"]
            if version:
                cmd.extend(["--branch", version])
            cmd.extend([url, str(dest)])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"  [Git] 克隆失败: {result.stderr.strip()}")
                return False
            git_dir = dest / ".git"
            if git_dir.exists():
                shutil.rmtree(git_dir, ignore_errors=True)
            return True
        except FileNotFoundError:
            print("  [Git] 未找到 git 命令，请先安装 Git")
            return False
        except subprocess.TimeoutExpired:
            print("  [Git] 克隆超时")
            return False
        except Exception as e:
            print(f"  [Git] 安装失败: {e}")
            return False


class PathInstaller:
    @staticmethod
    def install(source_path: str, dest: Path) -> bool:
        src = Path(source_path).resolve()
        if not src.exists():
            print(f"  [路径] 源路径不存在: {src}")
            return False
        if src.is_file() and src.suffix == ".zip":
            extract_zip(src, dest)
            return True
        if src.is_dir():
            manifest_path = src / MANIFEST_FILE
            if not manifest_path.exists():
                print(f"  [路径] 目录中缺少 {MANIFEST_FILE}: {src}")
                return False
            dest.mkdir(parents=True, exist_ok=True)
            for f in src.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(src)
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, target)
            return True
        print(f"  [路径] 不支持的源类型: {src}")
        return False


class DownloadManager:
    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def download_parallel(
        self,
        tasks: list[tuple[str, str, Path, str]],
    ) -> dict[str, Path]:
        results: dict[str, Path] = {}
        errors: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for name, version, cache_dir, registry_url in tasks:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached = cache_dir / f"{name}-{version}.zip"
                if cached.exists():
                    results[name] = cached
                    continue
                future = executor.submit(
                    self._download_one, name, version, cache_dir, registry_url
                )
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    path = future.result()
                    if path:
                        results[name] = path
                    else:
                        errors[name] = "下载失败"
                except Exception as e:
                    errors[name] = str(e)

        if errors:
            for name, err in errors.items():
                print(f"  [下载] {name}: {err}")

        return results

    def _download_one(
        self, name: str, version: str, cache_dir: Path, registry_url: str
    ) -> Path | None:
        try:
            if registry_url.startswith("local:"):
                local_path = registry_url[6:]
                registry = LocalRegistry(base_dir=Path(local_path))
                return registry.download_package(name, version, cache_dir)
            else:
                registry = PackageRegistry(registry_url)
                return registry.download_package(name, version, cache_dir)
        except (URLError, HTTPError, FileNotFoundError) as e:
            print(f"  [下载] {name}@{version} 从 {registry_url} 失败: {e}")
            return None


class PackageManager:
    def __init__(
        self,
        project_dir: Path | None = None,
        registry_url: str = DEFAULT_REGISTRY,
    ):
        self.project_dir = project_dir or Path.cwd()
        self.packages_dir = self.project_dir / PACKAGES_DIR
        self.manifest_path = self.project_dir / MANIFEST_FILE
        self.lock_path = self.project_dir / LOCK_FILE
        self.local_registry = LocalRegistry()
        self.registry_manager = RegistryManager(local_registry=self.local_registry)
        if registry_url != DEFAULT_REGISTRY:
            self.registry_manager.add_registry("custom", registry_url, priority=-1)
        self.resolver = VersionResolver(self.registry_manager, self.local_registry)
        self.download_manager = DownloadManager()

    def _ensure_dirs(self) -> None:
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        GLOBAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def init_project(
        self,
        name: str,
        author: str = "",
        description: str = "",
    ) -> PackageManifest:
        manifest = PackageManifest(
            name=name,
            version="0.1.0",
            description=description,
            author=author,
            main=f"{name}.nl",
        )
        manifest.save(self.manifest_path)
        self._ensure_dirs()
        nl_file = self.project_dir / manifest.main
        if not nl_file.exists():
            nl_file.write_text(
                f"# {name}\n# {description}\n\n定义函数 主函数():\n    输出(\"你好，世界！\")\n",
                encoding="utf-8",
            )
        print(f"已初始化项目: {name}")
        return manifest

    def load_manifest(self) -> PackageManifest:
        return PackageManifest.load(self.manifest_path)

    def save_manifest(self, manifest: PackageManifest) -> None:
        manifest.save(self.manifest_path)

    def load_lockfile(self) -> LockFile:
        return LockFile.load(self.lock_path)

    def save_lockfile(self, lockfile: LockFile) -> None:
        lockfile.save(self.lock_path)

    def _parse_package_spec(self, spec: str) -> tuple[str, str, str]:
        if spec.startswith("git+") or (spec.startswith("https://") and ".git" in spec):
            url, version = parse_git_spec(spec)
            name = Path(url).stem.replace(".git", "")
            return name, version, "git"
        if spec.startswith("file://") or (
            not spec.startswith("@") and (
                Path(spec).exists() or Path(spec).suffix == ".zip"
            )
        ):
            path = spec[7:] if spec.startswith("file://") else spec
            p = Path(path)
            name = p.stem.replace(".zip", "") if p.suffix == ".zip" else p.name
            return name, "", "path"
        if "@" in spec and not spec.startswith("@"):
            parts = spec.rsplit("@", 1)
            return parts[0], parts[1], "registry"
        return spec, "*", "registry"

    def install(
        self,
        package_names: list[str] | None = None,
        dev: bool = False,
        local: bool = False,
    ) -> LockFile:
        self._ensure_dirs()
        manifest = self.load_manifest() if self.manifest_path.exists() else PackageManifest()
        lockfile = self.load_lockfile()

        git_tasks: list[tuple[str, str, str]] = []
        path_tasks: list[tuple[str, str]] = []
        registry_deps: dict[str, str] = {}

        if package_names:
            for pkg_spec in package_names:
                name, version, source = self._parse_package_spec(pkg_spec)
                if source == "git":
                    git_tasks.append((name, version, pkg_spec))
                    if dev:
                        manifest.dev_dependencies[name] = pkg_spec
                    else:
                        manifest.dependencies[name] = pkg_spec
                elif source == "path":
                    path_tasks.append((name, pkg_spec))
                    if dev:
                        manifest.dev_dependencies[name] = pkg_spec
                    else:
                        manifest.dependencies[name] = pkg_spec
                else:
                    if dev:
                        manifest.dev_dependencies[name] = version
                    else:
                        manifest.dependencies[name] = version
                    registry_deps[name] = version
            self.save_manifest(manifest)

        all_deps = dict(manifest.dependencies)
        if dev or not package_names:
            all_deps.update(manifest.dev_dependencies)

        for name, vr in all_deps.items():
            _, _, source = self._parse_package_spec(vr)
            if source == "registry" and name not in registry_deps:
                registry_deps[name] = vr

        for name, version, git_spec in git_tasks:
            print(f"  安装 {name}@{version or 'latest'} (git)...")
            self._install_from_git(name, version, git_spec, lockfile)

        for name, path_spec in path_tasks:
            print(f"  安装 {name} (path)...")
            self._install_from_path(name, path_spec, lockfile)

        if registry_deps:
            self.resolver.clear()
            resolved = self.resolver.resolve(registry_deps)

            download_tasks: list[tuple[str, str, Path, str]] = []
            for name, version in resolved.items():
                lock_entry = lockfile.packages.get(name)
                if lock_entry and lock_entry.version == version:
                    pkg_dir = self.packages_dir / name
                    if pkg_dir.exists():
                        print(f"  {name}@{version} (已安装)")
                        continue

                node = self.resolver.get_resolution_tree().get(name)
                source_url = node.source if node else (
                    "local" if local else self.registry_manager.get_primary().url
                )

                cache_dir = GLOBAL_CACHE_DIR / name / version
                cached_tarball = cache_dir / f"{name}-{version}.zip"

                if cached_tarball.exists():
                    print(f"  安装 {name}@{version} (缓存)...")
                    self._install_from_cache(name, version, cached_tarball, lockfile, source_url)
                else:
                    download_tasks.append((name, version, cache_dir, source_url))

            if download_tasks:
                print(f"  下载 {len(download_tasks)} 个包...")
                downloaded = self.download_manager.download_parallel(download_tasks)
                for name, tarball in downloaded.items():
                    version = resolved.get(name, "")
                    node = self.resolver.get_resolution_tree().get(name)
                    source_url = node.source if node else self.registry_manager.get_primary().url
                    self._install_from_cache(name, version, tarball, lockfile, source_url)

            for name in list(lockfile.packages.keys()):
                if name not in resolved and lockfile.packages[name].source == "registry":
                    self._uninstall_package(name)
                    del lockfile.packages[name]
                    print(f"  移除 {name} (不再需要)")

        if self.resolver.conflicts:
            print(f"\n[警告] 发现 {len(self.resolver.conflicts)} 个依赖冲突:")
            for c in self.resolver.conflicts:
                print(f"  - {c.message}")

        self.save_lockfile(lockfile)
        self.save_manifest(manifest)
        total = len(lockfile.packages)
        print(f"安装完成: {total} 个包")
        return lockfile

    def _install_from_cache(
        self, name: str, version: str, tarball: Path,
        lockfile: LockFile, source_url: str,
    ) -> None:
        integrity = compute_hash(tarball)
        pkg_dir = self.packages_dir / name
        if pkg_dir.exists():
            shutil.rmtree(pkg_dir)
        pkg_dir.mkdir(parents=True, exist_ok=True)
        extract_zip(tarball, pkg_dir)

        if source_url.startswith("local:"):
            resolved_url = source_url
        else:
            resolved_url = f"{source_url}/api/packages/{name}/{version}/download"

        lockfile.packages[name] = LockEntry(
            name=name,
            version=version,
            resolved=resolved_url,
            integrity=integrity,
            source="registry",
        )

    def _install_from_git(
        self, name: str, version: str, git_spec: str, lockfile: LockFile,
    ) -> None:
        pkg_dir = self.packages_dir / name
        if pkg_dir.exists():
            shutil.rmtree(pkg_dir)
        url, _ = parse_git_spec(git_spec)
        success = GitInstaller.install(url, version, pkg_dir)
        if success:
            lockfile.packages[name] = LockEntry(
                name=name,
                version=version or "0.0.0-git",
                resolved=git_spec,
                integrity="",
                source="git",
            )
        else:
            print(f"  [警告] Git 安装 {name} 失败，跳过")

    def _install_from_path(
        self, name: str, path_spec: str, lockfile: LockFile,
    ) -> None:
        pkg_dir = self.packages_dir / name
        if pkg_dir.exists():
            shutil.rmtree(pkg_dir)
        source_path = path_spec[7:] if path_spec.startswith("file://") else path_spec
        success = PathInstaller.install(source_path, pkg_dir)
        if success:
            manifest_path = pkg_dir / MANIFEST_FILE
            version = "0.0.0-path"
            if manifest_path.exists():
                m = PackageManifest.load(manifest_path)
                version = m.version
            lockfile.packages[name] = LockEntry(
                name=name,
                version=version,
                resolved=path_spec,
                integrity="",
                source="path",
            )
        else:
            print(f"  [警告] 路径安装 {name} 失败，跳过")

    def uninstall(self, package_names: list[str]) -> None:
        if not self.manifest_path.exists():
            print("错误: 当前目录不是NLASM项目")
            return

        manifest = self.load_manifest()
        lockfile = self.load_lockfile()

        for name in package_names:
            if name in manifest.dependencies:
                del manifest.dependencies[name]
            if name in manifest.dev_dependencies:
                del manifest.dev_dependencies[name]
            self._uninstall_package(name)
            if name in lockfile.packages:
                del lockfile.packages[name]
            print(f"已卸载: {name}")

        self.save_manifest(manifest)
        self.save_lockfile(lockfile)

    def _uninstall_package(self, name: str) -> None:
        pkg_dir = self.packages_dir / name
        if pkg_dir.exists():
            shutil.rmtree(pkg_dir)

    def publish(self, local: bool = False) -> None:
        if not self.manifest_path.exists():
            print("错误: 当前目录不是NLASM项目")
            return

        manifest = self.load_manifest()
        if not manifest.name:
            print("错误: nlasm.json 中缺少 name 字段")
            return
        if not manifest.version:
            print("错误: nlasm.json 中缺少 version 字段")
            return

        if local:
            self.local_registry.publish(self.project_dir, manifest)
            print(f"已发布到本地仓库: {manifest.name}@{manifest.version}")
            return

        files_to_pack: list[Path] = []
        for pattern in ["**/*.nl", MANIFEST_FILE]:
            for f in sorted(self.project_dir.glob(pattern)):
                if PACKAGES_DIR in f.parts:
                    continue
                if f.name == LOCK_FILE:
                    continue
                files_to_pack.append(f)

        tmp_dir = Path(tempfile.mkdtemp(prefix="nlasm-pub-"))
        try:
            tarball = tmp_dir / f"{manifest.name}-{manifest.version}.zip"
            with zipfile.ZipFile(tarball, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in files_to_pack:
                    zf.write(f, f.relative_to(self.project_dir))

            primary = self.registry_manager.get_primary()
            url = f"{primary.url}/api/packages"
            data = json.dumps(manifest.to_dict(), ensure_ascii=False).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if primary.auth_token:
                headers["Authorization"] = f"Bearer {primary.auth_token}"
            req = Request(url, data=data, headers=headers, method="POST")
            try:
                with urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                print(f"已发布: {manifest.name}@{manifest.version}")
                if "message" in result:
                    print(f"  {result['message']}")
            except (URLError, HTTPError) as e:
                print(f"发布失败: {e}")
                print(f"  提示: 可以使用 --local 发布到本地仓库")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def list_packages(self) -> list[dict[str, str]]:
        if not self.packages_dir.exists():
            return []
        result: list[dict[str, str]] = []
        for pkg_dir in sorted(self.packages_dir.iterdir()):
            if not pkg_dir.is_dir():
                continue
            manifest_path = pkg_dir / MANIFEST_FILE
            if manifest_path.exists():
                m = PackageManifest.load(manifest_path)
                lock_entry = self.load_lockfile().packages.get(m.name or pkg_dir.name)
                result.append({
                    "name": m.name or pkg_dir.name,
                    "version": m.version,
                    "description": m.description,
                    "source": lock_entry.source if lock_entry else "unknown",
                })
            else:
                result.append({
                    "name": pkg_dir.name,
                    "version": "?",
                    "description": "",
                    "source": "unknown",
                })
        return result

    def outdated(self) -> list[dict[str, str]]:
        if not self.manifest_path.exists():
            return []
        manifest = self.load_manifest()
        lockfile = self.load_lockfile()
        result: list[dict[str, str]] = []
        all_deps = dict(manifest.dependencies)
        all_deps.update(manifest.dev_dependencies)

        for name, version_range in all_deps.items():
            lock_entry = lockfile.packages.get(name)
            if not lock_entry:
                continue
            current = lock_entry.version
            try:
                version, _ = self.registry_manager.resolve_version(name, version_range)
                if version != current:
                    result.append({
                        "name": name,
                        "current": current,
                        "latest": version,
                    })
            except (ValueError, FileNotFoundError, RuntimeError):
                pass
        return result

    def update(self, package_names: list[str] | None = None) -> LockFile:
        self._ensure_dirs()
        if not self.manifest_path.exists():
            print("错误: 当前目录不是NLASM项目")
            return LockFile()
        manifest = self.load_manifest()
        lockfile = self.load_lockfile()
        all_deps = dict(manifest.dependencies)
        all_deps.update(manifest.dev_dependencies)

        if package_names:
            targets = {n: all_deps.get(n, "*") for n in package_names if n in all_deps}
        else:
            targets = all_deps

        registry_targets: dict[str, str] = {}
        for name, vr in targets.items():
            _, _, source = self._parse_package_spec(vr)
            if source == "registry":
                registry_targets[name] = vr

        if registry_targets:
            self.resolver.clear()
            resolved = self.resolver.resolve(registry_targets)

            for name, version in resolved.items():
                lock_entry = lockfile.packages.get(name)
                if lock_entry and lock_entry.version != version:
                    print(f"  更新 {name}: {lock_entry.version} -> {version}")
                    node = self.resolver.get_resolution_tree().get(name)
                    source_url = node.source if node else self.registry_manager.get_primary().url
                    cache_dir = GLOBAL_CACHE_DIR / name / version
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cached = cache_dir / f"{name}-{version}.zip"
                    if cached.exists():
                        self._install_from_cache(name, version, cached, lockfile, source_url)
                    else:
                        try:
                            if source_url.startswith("local:"):
                                local_path = source_url[6:]
                                tarball = LocalRegistry(base_dir=Path(local_path)).download_package(name, version, cache_dir)
                            else:
                                registry = PackageRegistry(source_url)
                                tarball = registry.download_package(name, version, cache_dir)
                            self._install_from_cache(name, version, tarball, lockfile, source_url)
                        except (URLError, HTTPError, FileNotFoundError) as e:
                            print(f"  [警告] 更新 {name} 失败: {e}")
                elif not lock_entry:
                    print(f"  安装 {name}@{version}")

        self.save_lockfile(lockfile)
        return lockfile

    def run_script(self, script_name: str) -> None:
        if not self.manifest_path.exists():
            print("错误: 当前目录不是NLASM项目")
            return
        manifest = self.load_manifest()
        if script_name not in manifest.scripts:
            print(f"错误: 找不到脚本 '{script_name}'")
            return
        cmd = manifest.scripts[script_name]
        print(f"运行脚本: {cmd}")
        os.system(cmd)

    def info(self, package_name: str) -> dict[str, Any] | None:
        try:
            return self.registry_manager.fetch_package_info(package_name)
        except (FileNotFoundError, RuntimeError):
            return None

    def verify_integrity(self) -> list[str]:
        lockfile = self.load_lockfile()
        errors: list[str] = []
        for name, entry in lockfile.packages.items():
            pkg_dir = self.packages_dir / name
            if not pkg_dir.exists():
                errors.append(f"{name}: 包目录不存在")
                continue
            manifest_path = pkg_dir / MANIFEST_FILE
            if not manifest_path.exists():
                errors.append(f"{name}: 缺少包配置文件")
                continue
            m = PackageManifest.load(manifest_path)
            if m.version != entry.version:
                errors.append(
                    f"{name}: 版本不匹配 (锁定: {entry.version}, 实际: {m.version})"
                )
            if entry.integrity:
                cache_dir = GLOBAL_CACHE_DIR / name / entry.version
                cached = cache_dir / f"{name}-{entry.version}.zip"
                if cached.exists():
                    if not verify_integrity(cached, entry.integrity):
                        errors.append(f"{name}: 校验和不匹配 (可能被篡改)")
        return errors

    def dependency_tree(self) -> str:
        if not self.manifest_path.exists():
            return "错误: 当前目录不是NLASM项目"
        manifest = self.load_manifest()
        lockfile = self.load_lockfile()
        all_deps = dict(manifest.dependencies)
        all_deps.update(manifest.dev_dependencies)

        self.resolver.clear()
        self.resolver.resolve(all_deps)
        resolved = {name: entry.version for name, entry in lockfile.packages.items()}
        return self.resolver.print_dependency_tree(all_deps, resolved)

    def add_registry(self, name: str, url: str, **kwargs: Any) -> None:
        self.registry_manager.add_registry(name, url, **kwargs)
        print(f"已添加注册表: {name} ({url})")

    def remove_registry(self, name: str) -> None:
        if self.registry_manager.remove_registry(name):
            print(f"已移除注册表: {name}")
        else:
            print(f"注册表不存在: {name}")

    def list_registries(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.registry_manager.list_registries()]
