import pandas as pd
import numpy as np
import contextlib
import io
import re
from os.path import dirname, join


DEVICE_PREPROCESSED_DIR = "/data/local/tmp/llama/preprocessed"


def _ensure_year_col(df):
    if not isinstance(df, pd.DataFrame):
        return df
    if "year" in df.columns:
        return df
    if "Timestamp" not in df.columns:
        return df
    out_df = df.copy()
    out_df["year"] = pd.to_datetime(out_df["Timestamp"], errors="coerce").dt.year
    return out_df


def _sanitize_generated_code(code):
    code = code.replace("<code>", "")
    code = code.replace("</code>", "")
    code = code.replace("```python", "")
    code = code.replace("```", "")

    # Merge accidentally split identifiers, e.g. states_ data -> states_data.
    for _ in range(3):
        repaired = re.sub(r"([A-Za-z0-9])_\s+([A-Za-z0-9])", r"\1_\2", code)
        if repaired == code:
            break
        code = repaired

    # Repair LLM formatting glitches where assignments are split across lines.
    code = re.sub(
        r"(?m)^(\s*[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\n\s*([^\n#].*)$",
        r"\1 = \2",
        code,
    )

    # Fix malformed identifiers like avg_pm2.5 -> avg_pm2_5 (outside strings).
    code = re.sub(
        r"(?<![\"'])\b([A-Za-z_][A-Za-z0-9_]*)\.([0-9]+)\b",
        r"\1_\2",
        code,
    )

    # Normalize common relative pickle paths to device absolute paths.
    code = re.sub(
        r"([\"\'])(?:\.?/)?preprocessed[/\\](main_data\.pkl|ncap_funding_data\.pkl|states_data\.pkl)\1",
        rf"\1{DEVICE_PREPROCESSED_DIR}/\2\1",
        code,
    )
    code = re.sub(
        r"([\"\'])(?:\.?/)?preprocessed[/\\]ncap_data\.pkl\1",
        rf"\1{DEVICE_PREPROCESSED_DIR}/ncap_funding_data.pkl\1",
        code,
    )
    code = re.sub(
        r"([\"\'])(main_data\.pkl|ncap_funding_data\.pkl|states_data\.pkl)\1",
        rf"\1{DEVICE_PREPROCESSED_DIR}/\2\1",
        code,
    )
    code = re.sub(
        r"([\"\'])ncap_data\.pkl\1",
        rf"\1{DEVICE_PREPROCESSED_DIR}/ncap_funding_data.pkl\1",
        code,
    )

    # Fix a known malformed generated signature typo.
    code = code.replace(
        "def get_response(data: pd.DataFrame, states_ data: pd.DataFrame, ncap_ funding_ data: pd.DataFrame):",
        "def get_response(main_data: pd.DataFrame, states_data: pd.DataFrame, ncap_funding_data: pd.DataFrame):",
    )

    # Keep year naming consistent for downstream pivot(columns="year") logic.
    code = re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*[\"']Timestamp[\"']\s*\]\.dt\.year",
        r"_ensure_year_col(\1)['year']",
        code,
    )

    # Preserve columns later used after merge in generated UT variance queries.
    if (
        "population Density" in code
        and "['population']" in code
        and "['area (km2)']" in code
    ):
        code = code.replace(
            "[['state', 'population Density']]",
            "[['state', 'population Density', 'population', 'area (km2)']]",
        )
        code = code.replace(
            '[["state", "population Density"]]',
            '[["state", "population Density", "population", "area (km2)"]]'
        )
    if (
        "Population Density" in code
        and "['population']" in code
        and "['area (km2)']" in code
    ):
        code = code.replace(
            "[['state', 'Population Density']]",
            "[['state', 'Population Density', 'population', 'area (km2)']]",
        )
        code = code.replace(
            '[["state", "Population Density"]]',
            '[["state", "Population Density", "population", "area (km2)"]]'
        )
    return code


def _load_pickle(filename):
    candidates = [
        join(DEVICE_PREPROCESSED_DIR, filename),
        join(dirname(__file__), filename),
    ]
    last_error = None
    for path in candidates:
        try:
            return pd.read_pickle(path)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Unable to load {filename} from {DEVICE_PREPROCESSED_DIR} or module directory: {last_error}"
    )


def _prepare_states_data(states_data):
    if not isinstance(states_data, pd.DataFrame):
        return states_data

    prepared = states_data.copy()
    if "population" in prepared.columns and "area (km2)" in prepared.columns:
        population = pd.to_numeric(prepared["population"], errors="coerce")
        area = pd.to_numeric(prepared["area (km2)"], errors="coerce")
        density = population / area.replace(0, np.nan)
        for density_col in [
            "Population Density",
            "population Density",
            "population density",
            "population_density",
        ]:
            if density_col not in prepared.columns:
                prepared[density_col] = density

    return prepared


def _call_get_response(local_vars, main_data, states_data, ncap_data):
    get_response = local_vars.get("get_response")
    if not callable(get_response):
        return None

    call_styles = [
        lambda: get_response(main_data, states_data, ncap_data),
        lambda: get_response(
            main_data=main_data,
            states_data=states_data,
            ncap_funding_data=ncap_data,
        ),
        lambda: get_response(data=main_data, states_data=states_data, ncap_data=ncap_data),
    ]
    last_type_error = None
    for call_style in call_styles:
        try:
            return call_style()
        except TypeError as exc:
            last_type_error = exc
    if last_type_error is not None:
        raise last_type_error
    return None


def exec_code(code, mn_data, st_data, nc_data):
    local_vars = {}
    global_vars = {
        "__builtins__": __builtins__,
        "pd": pd,
        "np": np,
        "_ensure_year_col": _ensure_year_col,
        "main_data": mn_data,
        "states_data": st_data,
        "ncap_data": nc_data,
        "ncap_funding_data": nc_data,
    }

    stdout_buffer = io.StringIO()
    original_read_pickle = pd.read_pickle

    def _safe_read_pickle(path, *args, **kwargs):
        path_text = str(path).replace("\\", "/")
        file_name = path_text.split("/")[-1].lower().replace(" ", "").replace("_", "")
        if file_name == "maindata.pkl":
            return mn_data.copy()
        if file_name in ("statesdata.pkl", "states_data.pkl"):
            return st_data.copy()
        if file_name in ("ncapfundingdata.pkl", "ncapdata.pkl"):
            return nc_data.copy()
        return original_read_pickle(path, *args, **kwargs)

    pd.read_pickle = _safe_read_pickle
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, global_vars, local_vars)
    except Exception as e:
        return f"Error while executing code string: {e}"
    finally:
        pd.read_pickle = original_read_pickle

    captured_stdout = stdout_buffer.getvalue().strip()

    try:
        result = _call_get_response(local_vars, mn_data, st_data, nc_data)
        if result is not None:
            return result
    except Exception as e:
        return f"Error while executing get_response: {e}"

    if captured_stdout:
        return captured_stdout

    if "result" in local_vars and local_vars["result"] is not None:
        return local_vars["result"]

    return "Code executed successfully."


def main(code):
    main_data = _load_pickle("main_data.pkl")
    ncap_data = _load_pickle("ncap_funding_data.pkl")
    states_data = _load_pickle("states_data.pkl")
    states_data = _prepare_states_data(states_data)
    code = _sanitize_generated_code(code)
    output = exec_code(code, main_data, states_data, ncap_data)
    return output

