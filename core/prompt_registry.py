import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    version: str
    path: str
    description: str
    template: str

    def render(self, **variables: Any) -> str:
        return self.template.format(**variables)

    def metadata(self) -> Dict[str, str]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_version": self.version,
            "prompt_path": self.path,
        }


class PromptRegistry:
    """Loads prompt templates from a versioned manifest.

    Prompt text lives in plain files so changes are easy to diff, review, and
    roll back. `prompts/manifest.json` selects the active version per prompt.
    """

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        prompts_root: Optional[str] = None,
    ):
        project_root = Path(__file__).resolve().parent.parent
        self.prompts_root = Path(prompts_root) if prompts_root else project_root / "prompts"
        self.manifest_path = Path(manifest_path) if manifest_path else self.prompts_root / "manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def get(self, prompt_id: str, version: Optional[str] = None) -> PromptTemplate:
        prompt_config = self._manifest["prompts"][prompt_id]
        selected_version = version or prompt_config["active_version"]
        version_config = prompt_config["versions"][selected_version]
        prompt_path = version_config["path"]
        template_path = self.prompts_root / prompt_path

        with template_path.open("r", encoding="utf-8") as fp:
            template = fp.read()

        return PromptTemplate(
            prompt_id=prompt_id,
            version=selected_version,
            path=prompt_path,
            description=prompt_config.get("description", ""),
            template=template,
        )
