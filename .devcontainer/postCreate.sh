curl https://raw.githubusercontent.com/GitAlias/gitalias/master/gitalias.txt -o ~/.gitalias
git config --global include.path ~/.gitalias
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ruff ipython tensorboard torch-tb-profiler pytest
pip install -r requirements.txt