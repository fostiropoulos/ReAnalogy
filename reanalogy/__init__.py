from pathlib import Path
from reanalogy.dataset import ReAnalogy
package_dir = Path(__file__).parent.parent
data_path = package_dir.joinpath("data")
data_path.mkdir(exist_ok=True)