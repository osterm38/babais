# babais

This is a project that attempts to emulate the game "Baba Is You."


# user quick start

## install our (sub)package and cli globally

with `uv`
```sh
uv tool install "babais @ git+https://github.com/osterm38/babais.git"
# uv tool install "babais-web @ git+https://github.com/osterm38/babais.git#subdirectory=packages/babais-web"
```

or with `virtualenv` and `pip`
```sh
virtualenv .venv
source .venv/bin/activate
pip install "babais @ git+https://github.com/osterm38/babais.git"
# pip install "babais-web @ git+https://github.com/osterm38/babais.git#subdirectory=packages/babais-web"
```

## run cli

with `uv`
```sh
uv run babais --help
# uv run babais-web --help
```

or more directly
```sh
source .venv/bin/activate
babais --help
# babais-web --help
```


# dev quick start

## acquire source code

```sh
mkdir babais
cd babais
git clone https://github.com/osterm38/babais.git
```

## setup virtual environment and install our package

with `uv`
```sh
uv sync
```

or with `virtualenv` and `pip`
```sh
virtualenv .venv
source .venv/bin/activate
pip install -e .
```


# features

Major work in progress:

- babais-app: (web) app that launches the interactive game
  - babais-play: screen for playing the game
  - babais-create: screen for creating new levels
  - babais-watch: screen for watching a solution be constructed
- babais-alg: algorithms for making the gameplay functional
  - babais-logic: logic-based algorithms for finding a solution to a level
  - babais-learn: learners that play the game and figure out solutions without the logic-based algorithms
