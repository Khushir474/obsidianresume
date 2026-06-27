import datetime
import pathlib
import re


def generate_run_id(logs_folder: str) -> str:
    """Generate run_YYYYMMDD_NNN, incrementing NNN by scanning existing eval logs."""
    today = datetime.date.today().strftime("%Y%m%d")
    eval_dir = pathlib.Path(logs_folder) / "eval"
    pattern = re.compile(rf"^run_{today}_(\d{{3}})\.json$")

    counter = 0
    if eval_dir.exists():
        for f in eval_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                counter = max(counter, int(m.group(1)))

    return f"run_{today}_{counter + 1:03d}"
