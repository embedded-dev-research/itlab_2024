<!--  Modify README.md file in the actual repository for your needs -->

[![Build application](https://github.com/aobolensk/cpp-project-template/actions/workflows/main.yml/badge.svg)](https://github.com/aobolensk/cpp-project-template/actions/workflows/main.yml)
[![Static analysis](https://github.com/aobolensk/cpp-project-template/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/aobolensk/cpp-project-template/actions/workflows/static-analysis.yml)
[![CodeQL](https://github.com/aobolensk/cpp-project-template/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/aobolensk/cpp-project-template/actions/workflows/codeql-analysis.yml)


# C++ project template

This template is designed for quick start of C++ project.

## Build

```bash
$ git clone https://github.com/aobolensk/cpp-project-template --recusive
$ cd cpp-project-template
$ cmake -S . -B build
$ cmake --build build --parallel
```

## Test

```bash
$ build/bin/run_tests
```

## Code coverage

This project offers two methods to view code coverage:

* HTML artifacts with code test coverage details are available on each commit
* To configure [codecov.io](codecov.io) follow the official documentation. Token secrets: `CODECOV_TOKEN`
