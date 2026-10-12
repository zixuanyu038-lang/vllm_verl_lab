"""Resolve a model revision once, then use that same local snapshot everywhere."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

repo = "Qwen/Qwen2.5-0.5B-Instruct"
target = Path("models/qwen-0.5b").resolve()
revision = "7ae557604adf67be50417f59c2c2f167def9a775"
snapshot_download(repo_id=repo, revision=revision, local_dir=str(target))
Path("models/model_revision.json").write_text(
    json.dumps({"repo": repo, "revision": revision, "local_dir": str(target)}, indent=2),
    encoding="utf-8")
print(target)
