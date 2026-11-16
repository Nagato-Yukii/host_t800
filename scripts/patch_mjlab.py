"""Apply the upstream observation history patch to the selected Python environment."""
from pathlib import Path
import hashlib
import shutil
import mjlab
root=Path(__file__).resolve().parents[1]
source=root/'vendor/AMP_mjlab/mjlab_patch/mjlab/managers/observation_manager.py'
target=Path(mjlab.__file__).parent/'managers/observation_manager.py'
if source.read_bytes()!=target.read_bytes():
    backup=target.with_suffix('.py.before_host_t800')
    if not backup.exists():shutil.copy2(target,backup)
    shutil.copy2(source,target)
print('Observation patch ready:',target,hashlib.sha256(target.read_bytes()).hexdigest())
