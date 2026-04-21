import contextlib
import io
import os
import traceback


_PATH_ALIASES = ("preprcessed", "preprocessed", "preproccesed")


def _strip_code_fences(source: str) -> str:
    text = source.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if not lines:
        return text

    first_line = lines[0].strip()
    if first_line.startswith("```"):
        lines = lines[1:]

    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _with_aliases(path_text: str) -> list[str]:
    variants = {path_text}

    for alias in _PATH_ALIASES:
        forward_prefix = f"{alias}/"
        backward_prefix = f"{alias}\\"

        if path_text.startswith(forward_prefix):
            suffix = path_text[len(forward_prefix) :]
            for replacement in _PATH_ALIASES:
                variants.add(f"{replacement}/{suffix}")

        if path_text.startswith(backward_prefix):
            suffix = path_text[len(backward_prefix) :]
            for replacement in _PATH_ALIASES:
                variants.add(f"{replacement}\\{suffix}")

        forward_marker = f"/{alias}/"
        backward_marker = f"\\{alias}\\"

        if forward_marker in path_text:
            for replacement in _PATH_ALIASES:
                variants.add(path_text.replace(forward_marker, f"/{replacement}/"))

        if backward_marker in path_text:
            for replacement in _PATH_ALIASES:
                variants.add(path_text.replace(backward_marker, f"\\{replacement}\\"))

    return list(variants)


def _resolve_data_path(path_value, working_dir: str):
    try:
        path_text = os.fspath(path_value)
    except TypeError:
        return path_value

    if not isinstance(path_text, str):
        return path_value

    candidates = []
    if os.path.isabs(path_text):
        candidates.extend(_with_aliases(path_text))
    else:
        candidates.extend(_with_aliases(path_text))
        if working_dir:
            for variant in _with_aliases(path_text):
                candidates.append(os.path.join(working_dir, variant))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return path_value


def _patch_pandas_read_pickle(working_dir: str):
    try:
        import pandas as pd
    except Exception:
        return None

    original_read_pickle = pd.read_pickle

    def patched_read_pickle(path, *args, **kwargs):
        resolved_path = _resolve_data_path(path, working_dir)
        return original_read_pickle(resolved_path, *args, **kwargs)

    pd.read_pickle = patched_read_pickle
    return pd, original_read_pickle


def _restore_pandas_read_pickle(patch_state) -> None:
    if patch_state is None:
        return

    pd, original_read_pickle = patch_state
    pd.read_pickle = original_read_pickle


def run_code(source: str, working_dir: str = "") -> dict:
    code = _strip_code_fences(source)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    traceback_text = ""
    ok = True
    previous_working_dir = ""

    namespace = {"__name__": "__main__"}
    patch_state = None

    try:
        if working_dir and os.path.isdir(working_dir):
            previous_working_dir = os.getcwd()
            os.chdir(working_dir)

        patch_state = _patch_pandas_read_pickle(working_dir)

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            exec(compile(code, "<generated_code>", "exec"), namespace, namespace)
    except Exception:
        ok = False
        traceback_text = traceback.format_exc()
    finally:
        _restore_pandas_read_pickle(patch_state)
        if previous_working_dir:
            os.chdir(previous_working_dir)

    return {
        "ok": ok,
        "sanitizedCode": code,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
        "traceback": traceback_text,
    }
