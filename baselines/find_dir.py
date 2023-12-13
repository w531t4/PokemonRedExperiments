from pathlib import Path
from datetime import datetime
import re
import sys

def confirm_pokezip_presence(data: Path) -> bool:
    files = list(data.glob("*"))
    for each in files:
        if each.name.startswith("poke_") and each.name.endswith("_steps.zip"):
            return True
    return False

def get_session_date(session_name: str) -> datetime:
    data = re.search(r"^session_(?P<datestamp>\d{8})_(?P<timestamp>\d{4})_.*$", session_name)
    fields = data.groupdict()
    stamp = "%s%s" % (fields["datestamp"], fields["timestamp"])
    return datetime.strptime(stamp, "%Y%m%d%H%M")

def get_current_dir(basepath: Path) -> Path:
    dirs = basepath.glob("*")
    only_dirs = [x for x in dirs if x.is_dir()]
    only_dirs = filter(lambda b: re.match(r"^session_\d{8}_\d{4}_[0-9a-f]{8}$", b.name), only_dirs)
    only_dirs = filter(lambda b: confirm_pokezip_presence(b), only_dirs)
    return sorted(only_dirs, key=lambda z: get_session_date(session_name=z.name))[-1]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        result = get_current_dir(basepath=Path(__file__).parent / "sessions")
    else:
        result = get_current_dir(basepath=Path(str(sys.argv[1])))

    print(str(result))
