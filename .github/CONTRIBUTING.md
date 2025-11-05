
# Contributing to CodeEntropy

We welcome contributions from the community to improve and extend CodeEntropy. This guide outlines how to get started, make changes, and submit them for review.

---

## Getting Started

1. **Create a GitHub account**: [Sign up here](https://github.com/signup/free).
2. **Fork the repository**: [How to fork](https://help.github.com/articles/fork-a-repo/).
3. **Clone your fork locally**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/CodeEntropy.git
    cd CodeEntropy
    ```

4. **Create a virtual environment**:
    ```bash
    python -m venv codeentropy-dev
    source codeentropy-dev/bin/activate  # Linux/macOS
    codeentropy-dev\Scripts\activate     # Windows
    ```

5. **Install development dependencies**:
    ```bash
    pip install -e ".[testing,docs,pre-commit]"
    pre-commit install
    ```

---

## Making Changes

- **Use a feature branch**:
  ```bash
  git checkout -b 123-fix-levels
  ```
  You cannot commit directly to `main` as this is a protected branch.

- **Add your code**, documentation, and tests. All new features must include:
  - Unit tests
  - Documentation updates
  - Compliance with coding standards

- **Run tests**:
  ```bash
  pytest -v
  pytest --cov CodeEntropy --cov-report=term-missing
  ```

- **Run pre-commit checks**:
  These ensure formatting, linting, and basic validations.
  ```bash
  pre-commit run --all-files
  ```

---

## Submitting a Pull Request (PR)

1. Push your branch to GitHub.
2. Open a [pull request](https://help.github.com/articles/using-pull-requests/).
3. Use the templated Pull Request template to fill out:
   - A summary of what the PR is doing
   - List all the changes that the PR is proposing
   - Add how these changes will impact the repository
4. Ensure:
   - All tests pass
   - Pre-commit checks pass
   - Documentation is updated

5. Your PR will be reviewed by core developers. At least one approval is required before merging.

---

## Running Tests

- Full suite:
  ```bash
  pytest -v
  ```
- With coverage:
  ```bash
  pytest --cov CodeEntropy --cov-report=term-missing
  ```
- Specific module:
  ```bash
  pytest CodeEntropy/tests/test_CodeEntropy/test_levels.py
  ```
- Specific test:
  ```bash
  pytest CodeEntropy/tests/test_CodeEntropy/test_levels.py::test_select_levels
  ```

---

## Coding Standards

We use **pre-commit hooks** to enforce style and quality:

- **Black** for formatting
- **Isort** for import sorting
- **Flake8** for linting
- **Pre-commit-hooks** for:
  - Large file detection
  - AST validity
  - Merge conflict detection
  - YAML/TOML syntax checks

---

## Continuous Integration (CI)

All PRs trigger GitHub Actions to:

- Run tests
- Check style
- Build documentation
- Validate versioning

---

## Building Documentation

Build locally:
```bash
cd docs
make html
```

View in browser:
```
docs/build/html/index.html
```

Edit docs in:
- `docs/science.rst`
- `docs/developer_guide.rst`

---

## Reporting Issues

Found a bug or want a feature?

1. Open an issue on GitHub.
2. Include a clear description and input files if relevant.

---

## Additional Resources

- [GitHub Docs](https://help.github.com/)
- [PR Best Practices](http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/)
- [Contribution Guide](http://www.contribution-guide.org)
- [Thinkful PR Tutorial](http://www.thinkful.com/learn/github-pull-request-tutorial/#Time-to-Submit-Your-First-PR)
