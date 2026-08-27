"""Put src/ on the path once, so no test file needs a sys.path hack."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
