"""
数据集的环境接口
"""
import os
import importlib
import re
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
import yaml


_ENV_PATH = Path(__file__).parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


class Env:

    # 可被外部 prompt 使用的环境规则提示
    rule = (
        "你在 ALFWorld 文本环境中行动。每次只输出一个可执行的英文动作命令，"
        "例如: go to kitchen, open fridge, take apple from fridge, put apple in sink basin。"
    )

    _env = None
    _last_info: Dict[str, Any] = {}
    _last_done: bool = False
    _pending_obs: Any = None
    _pending_info: Dict[str, Any] = {}

    @classmethod
    def _expand_path(cls, path_value: Any) -> Any:
        if not isinstance(path_value, str):
            return path_value
        return os.path.expandvars(path_value)

    @classmethod
    def _data_root(cls) -> str:
        # 你当前机器上的默认位置；也可通过 ALFWORLD_DATA 覆盖
        return os.getenv("ALFWORLD_DATA", "/Users/richw/.cache/alfworld")

    @classmethod
    def _inject_dataset_paths(cls, config: Dict[str, Any]) -> None:
        data_root = cls._data_root()
        os.environ["ALFWORLD_DATA"] = data_root

        dataset = config.setdefault("dataset", {})
        logic = config.setdefault("logic", {})
        mask_rcnn = config.setdefault("mask_rcnn", {})
        env_cfg = config.setdefault("env", {})

        dataset.setdefault("data_path", "$ALFWORLD_DATA/json_2.1.1/train")
        dataset.setdefault("eval_id_data_path", "$ALFWORLD_DATA/json_2.1.1/valid_seen")
        dataset.setdefault("eval_ood_data_path", "$ALFWORLD_DATA/json_2.1.1/valid_unseen")
        dataset["data_path"] = cls._expand_path(dataset["data_path"])
        dataset["eval_id_data_path"] = cls._expand_path(dataset["eval_id_data_path"])
        dataset["eval_ood_data_path"] = cls._expand_path(dataset["eval_ood_data_path"])

        logic.setdefault("domain", "$ALFWORLD_DATA/logic/alfred.pddl")
        logic.setdefault("grammar", "$ALFWORLD_DATA/logic/alfred.twl2")
        logic["domain"] = cls._expand_path(logic["domain"])
        logic["grammar"] = cls._expand_path(logic["grammar"])

        mask_rcnn.setdefault("pretrained_model_path", "$ALFWORLD_DATA/detectors/mrcnn.pth")
        mask_rcnn["pretrained_model_path"] = cls._expand_path(mask_rcnn["pretrained_model_path"])

        env_cfg.setdefault("type", "AlfredTWEnv")

    @classmethod
    def _load_alfworld_config(cls) -> Dict[str, Any]:
        candidates: List[Path] = []

        config_path = os.getenv("ALFWORLD_CONFIG")
        if config_path:
            candidates.append(Path(config_path))

        # 优先项目内配置
        candidates.append(Path(__file__).resolve().parent / "alfworld_config.yaml")

        # 其次尝试包内默认路径（若存在）
        try:
            pkg = importlib.import_module("alfworld")
            candidates.append(Path(pkg.__file__).resolve().parent / "config.yaml")
        except Exception:
            pass

        for cfg_file in candidates:
            if cfg_file.exists():
                with cfg_file.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f)

        candidate_str = ", ".join(str(p) for p in candidates)
        raise RuntimeError(
            "未找到 ALFWorld 配置文件。"
            f"已尝试: {candidate_str}。"
            "请在 ReCAP/.env 设置 ALFWORLD_CONFIG。"
        )

    @classmethod
    def _ensure_env(cls) -> None:
        if cls._env is not None:
            return

        try:
            env_mod = importlib.import_module("alfworld.agents.environment")
            get_environment = getattr(env_mod, "get_environment")
        except ImportError as exc:
            raise RuntimeError(
                "未检测到 ALFWorld。请先安装: pip install 'alfworld[full]' "
                "并执行 alfworld-download 下载数据。"
            ) from exc

        config = cls._load_alfworld_config()
        cls._inject_dataset_paths(config)
        env_type = os.getenv("ALFWORLD_ENV_TYPE", config["env"]["type"])
        train_eval = os.getenv("ALFWORLD_SPLIT", "eval_out_of_distribution")
        cls._env = get_environment(env_type)(config, train_eval=train_eval).init_env(batch_size=1)

    @staticmethod
    def _pick_first(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return value[0] if value else ""
        return value

    @classmethod
    def _obs_to_text(cls, obs: Any) -> str:
        obs0 = cls._pick_first(obs)
        if isinstance(obs0, bytes):
            return obs0.decode("utf-8", errors="ignore")
        return str(obs0)

    @classmethod
    def _extract_query_from_obs(cls, obs_text: str) -> str:
        text = (obs_text or "").strip()
        if not text:
            return ""

        patterns = (
            r"(?i)your task is to[:\s]+(.+)",
            r"(?i)you are tasked with[:\s]+(.+)",
            r"(?i)objective[:\s]+(.+)",
            r"(?i)goal[:\s]+(.+)",
        )
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()
        return text.splitlines()[0].strip()

    @classmethod
    def _admissible_commands(cls) -> List[str]:
        cmds = cls._last_info.get("admissible_commands", [])
        if isinstance(cmds, (list, tuple)) and cmds:
            first = cmds[0]
            if isinstance(first, (list, tuple)):
                return [str(x) for x in first]
            return [str(x) for x in cmds]
        return []

    @classmethod
    def _normalize_action(cls, action: str) -> str:
        action = str(action).strip()
        if not action:
            return "look"
        admissible = cls._admissible_commands()
        if admissible and action not in admissible:
            # 保持系统健壮性：若动作不合法，退回到一个安全动作
            return "look"
        return action

    @classmethod
    def peek_query(cls) -> str:
        """
        预取一条数据样本的 query（不丢失该样本，会被下一次 reset 复用）。
        """
        cls._ensure_env()

        if cls._pending_obs is None:
            obs, info = cls._env.reset()
            cls._pending_obs = obs
            cls._pending_info = info if isinstance(info, dict) else {}

        obs_text = cls._obs_to_text(cls._pending_obs)
        return cls._extract_query_from_obs(obs_text)

    @classmethod
    def reset(cls, task: str) -> str:
        """
        重置环境，返回初始状态字符串
        """
        cls._ensure_env()
        if cls._pending_obs is not None:
            obs = cls._pending_obs
            info = cls._pending_info
            cls._pending_obs = None
            cls._pending_info = {}
        else:
            obs, info = cls._env.reset()
        cls._last_info = info if isinstance(info, dict) else {}
        cls._last_done = False

        obs_text = cls._obs_to_text(obs)
        # ALFWorld 的真实任务来自数据集；保留外部 task 作为提示，便于兼容上层输入。
        task = str(task).strip()
        if task:
            return f"{obs_text}\n[External task hint]: {task}"
        return obs_text

    @classmethod
    def step(cls, action: str) -> str:
        """
        执行动作并返回观察字符串（为兼容现有 nodes.py）。
        """
        cls._ensure_env()
        norm_action = cls._normalize_action(action)

        obs, scores, dones, infos = cls._env.step([norm_action])
        cls._last_done = bool(cls._pick_first(dones))
        cls._last_info = infos if isinstance(infos, dict) else {}

        obs_text = cls._obs_to_text(obs)
        reward = cls._pick_first(scores)
        done = cls._pick_first(dones)
        return f"{obs_text}\n[reward={reward}, done={done}]"
