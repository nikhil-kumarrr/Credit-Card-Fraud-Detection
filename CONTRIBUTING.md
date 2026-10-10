# Contributing

Thanks for taking a look. Bug reports, fixes and ideas are welcome.

## Setup

~~~
git clone https://github.com/nikhil-kumarrr/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
~~~

## Before opening a pull request

~~~
ruff format src tests
ruff check src tests
pytest --cov
~~~

All three must pass. The same checks run in GitHub Actions.

## Guidelines

- Keep pull requests small and focused on one change.
- Add or update a test for every bug fix and every new feature.
- Use type hints and short docstrings for new functions.
- Write commit messages in the imperative, for example `Fix threshold validation`.
- The deployed model in `models/` is never overwritten by training. `python -m fraud_detection.train` writes to `models/retrained`.