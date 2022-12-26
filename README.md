# interval-diff
Implementation of a vectorised interval set difference operation in numpy.

Consider two sets of intervals `A` and `B` such that
```
A     : (----)  (----)  (----)  (----)         (----) (------)
B     :    (---------------)      (------)  (----)      (----)
```

We'd like to calculate `A \ B` where `A \ B` is
```
A \ B : (--)               (-)  (-)              (--) ()
```

## Installation
Clone the repo to your system and install

```bash
$ git clone https://github.com/BlakeJC94/interval-diff
$ cd interval-diff
$ pip install .
```

## Quickstart
TODO

## Future developments
- [ ] Implement algorithm for pandas DataFrames

## Contributing
Pull requests are most welcome!

* Code is styled using `[black](https://github.com/psf/black)`
    * Included in dev requirements
* Code is linted with `pylint` (`pip install pylint`)
* Requirements are managed using `pip-tools` (run `pip install pip-tools` if needed)
    * Add dependencies by adding packages to `setup.py` and running
      `./scripts/compile-requirements.sh`
    * Add dev dependencies to `setup.py` under `extras_require` and run
      `./scripts/compile-requirements-dev.sh`
* [Semantic versioning](https://semver.org) is used in this repo (shockingly)
    * Major version: rare, substantial changes that break backward compatibility
    * Minor version: most changes - new features, models or improvements
    * Patch version: small bug fixes and documentation-only changes

Virtual environment handling by `pyenv` is preferred:
```bash
# in the project directory
$ pyenv virtualenv 3.9.7 interval-diff
$ pyenv local interval-diff
$ pip install -e .
```
