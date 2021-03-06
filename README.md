# <img src="https://twiki.cern.ch/twiki/pub/BEABP/Logos/OMC_logo.png" height="28"> 3

This is the python-tool package of the optics measurements and corrections group (OMC).

If you are not part of that group, you will most likely have no use for the codes provided here, 
unless you have a 9km wide accelerator at home.
Feel free to use them anyway, if you wish!

## Documentation

- Autogenerated docs via ``sphinx`` can be found on <https://pylhc.github.io/omc3/>.
- General documentation of the OMC-Teams software on <https://twiki.cern.ch/twiki/bin/view/BEABP/OMC>

## Getting Started

### Prerequisites

The codes use a multitude of packages as can be found in the [requirements.txt](requirements.txt).

Important ones are: ``numpy``, ``pandas`` and ``scipy``.

### Installing

This package is not deployed, hence you need to use the standard git-commands to get a local copy.

## Description

This is the new repository ([old one](https://github.com/pylhc/Beta-Beat.src)) of the codes,
rewritten for python 3.6+.  


## Functionality

### Changelog

- Still building up to first release.

### Implemented

- Main functions: hole_in_one, harpy

- Utils: logging, iotools, file handlers, entrypoint

- Madx wrapper


### Development in progress

- optics measurement

- accelerator class

- k-mod

- Accuracy tests

- Regression tests

## Quality checks

### Tests

- Pytest unit tests are run automatically after each commit via 
[Travis-CI](https://travis-ci.com/pylhc/omc3). 

### Maintainability

- Additional checks for code-complexity, design-rules, test-coverage, duplication on 
[CodeClimate](https://codeclimate.com/github/pylhc/omc3)

- Direct commits to master are forbidden.

- All pull requests need to be reviewed!


## Authors

* **pyLHC/OMC-Team** - *Working Group* - [pyLHC](https://github.com/orgs/pylhc/teams/omc-team)

<!--
## License
This project is licensed under the  License - see the [LICENSE.md](LICENSE.md) file for details
-->
