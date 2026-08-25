# Board Game Recommender

[![PyPI](https://img.shields.io/pypi/v/board-game-recommender?style=flat-square)](https://pypi.python.org/pypi/board-game-recommender/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/board-game-recommender?style=flat-square)](https://pypi.python.org/pypi/board-game-recommender/)
[![PyPI - License](https://img.shields.io/pypi/l/board-game-recommender?style=flat-square)](https://pypi.python.org/pypi/board-game-recommender/)
[![Coookiecutter - Wolt](https://img.shields.io/badge/cookiecutter-Wolt-00c2e8?style=flat-square&logo=cookiecutter&logoColor=D4AA00&link=https://github.com/woltapp/wolt-python-package-cookiecutter)](https://github.com/woltapp/wolt-python-package-cookiecutter)


---

**Documentation**: [https://recommend-games.github.io/board-game-recommender](https://recommend-games.github.io/board-game-recommender)

**Source Code**: [https://github.com/recommend-games/board-game-recommender](https://github.com/recommend-games/board-game-recommender)

**PyPI**: [https://pypi.org/project/board-game-recommender/](https://pypi.org/project/board-game-recommender/)

---

Board game recommendation engine.

## Installation

```sh
pip install board-game-recommender
```

## Development

* Clone this repository
* Requirements:
  * [Poetry](https://python-poetry.org/)
  * Python 3.12+
* Create a virtual environment and install the dependencies

```sh
poetry install
```

* Activate the virtual environment

```sh
poetry shell
```

### Testing

```sh
pytest
```

### Documentation

The documentation is automatically generated from the content of the [docs directory](https://github.com/recommend-games/board-game-recommender/tree/master/docs) and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github Pages page](https://pages.github.com/) automatically as part each release.

### Releasing

Trigger the [Draft release workflow](https://github.com/recommend-games/board-game-recommender/actions/workflows/draft_release.yml)
(press _Run workflow_). This will update the changelog & version and create a GitHub release which is in _Draft_ state.

Find the draft release from the
[GitHub releases](https://github.com/recommend-games/board-game-recommender/releases) and publish it. When
 a release is published, it'll trigger [release](https://github.com/recommend-games/board-game-recommender/blob/master/.github/workflows/release.yml) workflow which creates PyPI
 release and deploys updated documentation.

### Pre-commit

Pre-commit hooks run all the auto-formatting (`ruff format`), linters (e.g. `ruff` and `mypy`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```

Or if you want them to run only for each push:

```sh
pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pre-commit run --all-files
```

### Releasing manually

If you'd rather not use GitHub Actions or the `gh` CLI, you can cut a release entirely from the command line:

```sh
# 1. Bump version (patch|minor|major|prepatch|preminor|premajor|prerelease, or explicit e.g. 1.2.3)
poetry version patch
VERSION=$(poetry version --short)

# 2. Update changelog
poetry run kacl-cli release "$VERSION" --modify --auto-link

# 3. Commit the version bump
git add CHANGELOG.md pyproject.toml
git commit -m "Release $VERSION"

# 4. Tag and push. `master` tracks GitLab, but the docs are published from
#    GitHub, so both remotes need the commit and the tag.
git tag "$VERSION"
git push gitlab master
git push gitlab "$VERSION"
git push github master
git push github "$VERSION"

# 5. Build and publish to PyPI (needs a token, e.g. via a PYPI_TOKEN env var)
poetry config pypi-token.pypi "$PYPI_TOKEN"
poetry publish --build

# 6. Deploy docs to GitHub Pages
poetry run mkdocs gh-deploy --force --remote-name github
```

Three things worth knowing:

* Tags carry no `v` prefix — `4.0.0`, not `v4.0.0`. That is what the draft release
 workflow creates, and what the changelog's auto-generated links expect. The older
 `v3.6.0`-style tags predate this convention.
* `mkdocs gh-deploy` pushes to a remote called `origin` by default, which this
 repository does not have, hence `--remote-name github`.
* `poetry version` only edits `pyproject.toml`. `poetry.lock` records a hash of the
 dependencies rather than the project's own version, so it does not need committing.

Note that no GitHub Release object gets created this way — only the PyPI package and docs. Steps 4 and 6 use plain `git push` (not the GitHub API), so they work without `gh`.

---

This project was generated using the [wolt-python-package-cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter) template.
