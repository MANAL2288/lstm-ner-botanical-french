"""Build the self-contained WMC-only Colab notebook for Said and MS."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def md(s): return dict(cell_type='markdown', metadata={}, source=s.splitlines(True))
def code(s): return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s.splitlines(True))
cells = [md('''# FloraNER — WMC (Said & MS)
Run on a Colab T4 GPU. This notebook runs only the duo's baseline: seeds 1, 2, 3; 60 epochs maximum; patience 10; Adam lr=0.001; BiWMC H=128; dropout 0.3; batch 32. Same data pipeline and evaluation as Vanilla. Best validation epoch is evaluated on test. No collective comparison, report, or slides are generated.
'''), code('''import os, sys, subprocess
import torch
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU, then run again.'
print(torch.__version__, torch.cuda.get_device_name(0))
from google.colab import drive
drive.mount('/content/drive')
os.environ['FLORANER_RESULTS'] = '/content/drive/MyDrive/floraner_wmc_results_e60'
os.makedirs(os.environ['FLORANER_RESULTS'], exist_ok=True)
os.makedirs('/content/floraner/src', exist_ok=True)
os.makedirs('/content/floraner/tests', exist_ok=True)
os.chdir('/content/floraner')
def run(*args):
    subprocess.run([sys.executable, *args], check=True)
run('-m', 'pip', 'install', '-q', 'pytest', 'numpy')
''')]
for rel in ['src/data.py','src/download_data.py','src/metrics.py','src/cells.py','src/model.py','src/train.py','src/sweep.py','tests/test_model.py','tests/test_wmc.py','src/summarize_wmc.py']:
    cells.append(code('%%writefile '+rel+'\n'+(ROOT/rel).read_text(encoding='utf-8-sig')))
cells += [md('## Verify and prepare the common data'), code("run('-m', 'pytest', 'tests', '-q')\nrun('src/download_data.py')"), md('## Three full WMC runs\nCompleted runs are skipped on rerun. An interrupted run restarts from its seed; partial epochs are not checkpointed.'), code("run('-u', 'src/sweep.py', '--cell', 'wmc', '--study', 'baseline', '--seeds', '1', '2', '3', '--epochs', '60', '--patience', '10')"), md('## WMC results and handoff archive'), code("""run('src/summarize_wmc.py')
import shutil
from google.colab import files
archive = shutil.make_archive('/content/floraner_wmc_results', 'zip', os.environ['FLORANER_RESULTS'])
files.download(archive)
""")]
nb=dict(cells=cells, metadata={'accelerator':'GPU','colab':{'provenance':[],'gpuType':'T4'},'kernelspec':{'display_name':'Python 3','name':'python3'},'language_info':{'name':'python'}}, nbformat=4, nbformat_minor=0)
p=ROOT/'notebooks/FloraNER_WMC_colab.ipynb'
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print(p)
