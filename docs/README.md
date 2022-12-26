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
To use this algorithm, simply import and use it:
```python
>>> import numpy as np
>>> from interval_diff.vectorised import interval_difference
>>> intervals_a = np.array(
...     [
...         (100, 200),
...         (300, 400),
...         (500, 600),
...         (700, 800),
...         (1000, 1100),
...         (1250, 1400),
...     ]
... )
>>> intervals_b = np.array(
...     [
...         (150, 580),
...         (720, 890),
...         (930, 1070),
...         (1300, 1400),
...     ]
... )
>>> result = interval_difference(intervals_a, intervals_b)
>>> print(result)
# [[ 100.  150.]
#  [ 580.  600.]
#  [ 700.  720.]
#  [1070. 1100.]
#  [1250. 1300.]]
```

To visualise the intervals, the included plotly function can be used
```python
>>> from interval_diff.vis import plot_intervals
>>> fig = plot_intervals(
...     [intervals_a, intervals_b, result],
...     names=["A", "B", "A \ B"],
...     colors=["blue", "red", "blue"],
... )
>>> fig.show()
```

![Intervals figure](./docs/img.png)

To run a quick benchmark, use the CLI interface
```bash
$ interval-diff
100%|███████████████████████████████████████████████████████████████████████████████████| 21/21 [00:22]
----------------------------------------------------------------------------------------------------
N intervals (3 samples)         | Non-vectorised mean time (s)    | Vectorised mean runtime (s)
----------------------------------------------------------------------------------------------------
20                              | 0.000125                        | 0.000198
100                             | 0.001245                        | 0.000218
500                             | 0.016832                        | 0.000404
1000                            | 0.061689                        | 0.000576
2000                            | 0.239640                        | 0.001052
5000                            | 1.464468                        | 0.002527
10000                           | 5.855935                        | 0.005418
```
Results will be written to `results.csv`

Parameters can be configured like so:
```bash
$ interval-diff --n-intervals 10 20 30 --n-samples 400
100%|███████████████████████████████████████████████████████████████████████████████| 1200/1200 [00:00]
----------------------------------------------------------------------------------------------------
N intervals (400 samples)       | Non-vectorised mean time (s)    | Vectorised mean runtime (s)
----------------------------------------------------------------------------------------------------
10                              | 0.000045                        | 0.000109
20                              | 0.000083                        | 0.000109
30                              | 0.000135                        | 0.000114
```

## Future developments
- [ ] Implement algorithm for pandas DataFrames
- [ ] Refactor tests

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
* [Semantic versioning](https://semver.org) is used in this repo
    * Major version: rare, substantial changes that break backward compatibility
    * Minor version: most changes - new features, models or improvements
    * Patch version: small bug fixes and documentation-only changes

Virtual environment handling by `pyenv` is preferred. Run `./scripts/create-pyenv.sh` for a quickstart
