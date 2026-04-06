import re
from pathlib import Path

def get_checkpoints(directory: str) -> list[Path]:
    path = Path(directory)

    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"provided path {directory} is not a valid file or directory")
    
    checkpoint_pattern = re.compile(r"checkpoint_(\d+)\.pt")

    checkpoints = []
    for file in path.iterdir():
        if file.is_file():
            match = checkpoint_pattern.match(file.name)
            if match or file.name == "final.pt":
                checkpoints.append((int(match.group(1)), file))
    checkpoints.sort(key=lambda x: x[0])

    return [file for _, file in checkpoints]