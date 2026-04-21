import contextlib
import io
import os
import traceback


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


def run_code(source: str, working_dir: str = "") -> dict:
    code = _strip_code_fences(source)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    traceback_text = ""
    ok = True
    previous_working_dir = ""

    namespace = {"__name__": "__main__"}

    try:
        if working_dir and os.path.isdir(working_dir):
            previous_working_dir = os.getcwd()
            os.chdir(working_dir)

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            exec(compile(code, "<generated_code>", "exec"), namespace, namespace)
    except Exception:
        ok = False
        traceback_text = traceback.format_exc()
    finally:
        if previous_working_dir:
            os.chdir(previous_working_dir)

    return {
        "ok": ok,
        "sanitizedCode": code,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
        "traceback": traceback_text,
    }
