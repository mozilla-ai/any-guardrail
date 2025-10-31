# Contributing to mozilla.ai any-guardrail

Thank you for your interest in contributing to this `any-guardrail`!

We're building tools to help developers integrate AI safety guardrails into their projects using open-source models. Whether you're fixing a typo, adding a new guardrail, or improving our architecture, your help is appreciated.


## Quick Links

- 📋 [Code of Conduct](CODE_OF_CONDUCT.md)
- 🐛 [Report a Bug](https://github.com/mozilla-ai/any-guardrail/issues/new?template=bug_report.md)
- 💡 [Request a Feature](https://github.com/mozilla-ai/any-guardrail/issues/new?template=feature_request.md)
- 🆕 [Request a Guardrail](https://github.com/mozilla-ai/any-guardrail/issues/new?template=guardrail_request.md)
- 🎯 [Good First Issues](https://github.com/mozilla-ai/any-guardrail/labels/good-first-issue)
- 💬 [GitHub Discussions](https://github.com/mozilla-ai/any-guardrail/discussions)


## Before You Start

### Check for Duplicates

Before creating a new issue or starting work:
- [ ] Search [existing issues](https://github.com/mozilla-ai/any-guardrail/issues) for duplicates
- [ ] Check [open pull requests](https://github.com/mozilla-ai/any-guardrail/pulls) to see if someone is already working on it
- [ ] For bugs, verify it still exists in the `main` branch

### Discuss Major Changes First

For significant changes, please open an issue **before** starting work:

- New guardrail integrations
- API changes or new public methods
- Architectural changes
- Breaking changes
- New dependencies

This ensures alignment with project goals and saves everyone time. **Maintainers reserve the right to close issues and PRs that do not align with the library roadmap.**

### Read Our Code of Conduct

All contributors must follow Mozilla's [Community Participation Guidelines](https://www.mozilla.org/about/governance/policies/participation/). We're committed to maintaining a welcoming, inclusive community.

## Development Setup

### Prerequisites

- **Python 3.11 or newer**
- **Git**
- **uv** (or your preferred package manager)

### Quick Start

We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) as your Python package and project manager.

```bash
# 1. Fork the repository on GitHub
# Click the "Fork" button at https://github.com/mozilla-ai/any-guardrail

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/any-guardrail.git
cd any-guardrail

# 3. Add upstream remote
git remote add upstream https://github.com/mozilla-ai/any-guardrail.git

# 4. Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync --dev --extra all

# 5. Ensure all checks pass
pre-commit run --all-files

# 6. Verify your setup
pytest -v tests
```

## Making Changes

### 1. Create a Branch

Always work on a feature branch, never directly on `main`:

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `guardrail/` - New guardrail integrations
- `refactor/` - Code improvements without behavior changes

**Important**: Review issue discussion fully before starting work. If an issue is under discussion, engage in the thread first to ensure alignment.

### 2. Make Changes

Make your changes following our code quality standards:

#### Code Clarity and Style

- **Readability first**: Code must be self-documenting. If logic is non-obvious, include clear, concise comments. See [this guide](https://swimm.io/learn/documentation-tools/tips-for-creating-self-documenting-code) for tips.
- **Consistent style**: Follow existing codebase conventions (function naming, docstring format)
- **No dead code**: Remove commented-out blocks, leftover print statements, and unrelated refactors
- **Error handling**: Document failure modes and handle them with robust exception handling
- **Type hints**: Add type hints to all functions

### 3. Write Tests

**Every change needs tests!** This is non-negotiable.

#### Test Requirements

- **Coverage**: All new functionality must include unit tests covering both happy paths and relevant edge cases
- **No silent failures**: Tests should fail loudly on errors. No `assert True` placeholders
- **Passing tests**: All tests must pass locally before submitting PR

Run tests with:

```bash
pytest -v tests
```

### 4. Update Documentation

Documentation is as important as code!

Update when you:
- Add a new feature
- Change existing behavior
- Add a new guardrail
- Fix a bug that affects usage

Documentation to update:
- **Docstrings** in code (required)
- **README.md** if changing core functionality
- **docs/** for guardrail additions or feature changes

Preview documentation locally:

```bash
mkdocs serve
```

### 5. Commit Your Changes

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add support for toxic content detection guardrail"
git commit -m "Fix streaming response handling in base guardrail"
git commit -m "Update documentation for custom guardrail creation"

# Less helpful commit messages (avoid these)
git commit -m "fix bug"
git commit -m "update"
git commit -m "wip"
```

## Adding a New Guardrail

Adding guardrail support is a major contribution! Here's the complete process:

### 1. Check Requirements

Before requesting or implementing:

- [ ] Guardrail addresses a genuine safety or moderation need
- [ ] Model/approach is available via HuggingFace or has well-documented API
- [ ] Guardrail is actively maintained and supported
- [ ] No existing issue/PR for adding this guardrail
- [ ] Discussed via GitHub issue first

### 2. Implementation Checklist

Create your guardrail following this structure:

```
any_guardrail/
├── guardrails/
│   ├── 📂 your_guardrail/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 your_guardrail.py   # Main guardrail implementation
│   │   └── 📝 ...                 # Any extra files
```

**Required Implementation**:

- [ ] Create guardrail class in `src/any_guardrail/guardrails/`
- [ ] Inherit from `Guardrail` base class (or `HuggingFace` if applicable)
- [ ] Implement all abstract methods:
  - For general guardrails: methods defined in `Guardrail` base class
  - For HuggingFace models: `preprocessing`, `inference`, and `postprocessing` functions
- [ ] Add your guardrail to `GuardrailName` enum in appropriate file
  - Enum key should be `UPPER_CASE`
  - String value should be `lower_case`
- [ ] Handle guardrail-specific errors gracefully
- [ ] Add comprehensive type hints and docstrings
- [ ] Document failure modes explicitly

**Testing Requirements**:

- [ ] Unit tests for all guardrail functions
- [ ] Tests covering happy path and edge cases
- [ ] Error handling tests
- [ ] No silent test failures
- [ ] All tests pass locally

**Documentation Requirements**:

- [ ] Add to relevant documentation with usage examples
- [ ] Document limitations and failure modes
- [ ] Update installation instructions if needed

## Submitting Your Contribution

### 1. Ensure Quality Standards

Before submitting:

```bash
# Run linting
pre-commit run --all-files

# Run tests
pytest -v tests
```

### 2. Push to Your Fork

```bash
# Commit your changes
git add .
git commit -m "feat: add toxic content detection guardrail"

# Push to your fork
git push origin feature/your-feature
```

### 3. Create a Pull Request

1. Go to https://github.com/mozilla-ai/any-guardrail
2. Click "New Pull Request"
3. Click "compare across forks"
4. Select your fork and branch
5. Fill out the [PR template](pull_request_template.md) completely
6. Click "Create Pull Request"

## Review Process

### What to Expect

1. **Initial Response**: Within **5 business days**
2. **Simple Fixes**: Usually merged within **1 week**
3. **Complex Features**: May take **2-3 weeks** for thorough review
4. **Guardrail Integrations**: Often require **2-3 review cycles**

### During Review

- Maintainers will provide constructive feedback
- Address comments with new commits (don't force push)
- Ask questions if feedback is unclear
- Be patient and respectful
- All checks must pass before merge

### If Your PR Goes Stale

- No activity for **30+ days** may result in closure
- You can always reopen and continue later
- Let us know if you need help finishing

## Your First Contribution

New to open source? Welcome! Here's how to get started:

### Step 1: Find an Issue

Look for issues labeled:
- `good-first-issue` - Perfect for newcomers
- `help-wanted` - Community contributions welcome
- `documentation` - Often accessible for beginners

### Step 2: Claim the Issue

Comment on the issue:
> "Hi! I'd like to work on this. Is it still available?"

We'll assign it to you and provide guidance.

### Step 3: Ask Questions Early

Don't spend days stuck! Ask questions:
- In the issue comments
- In GitHub Discussions
- Tag maintainers if needed

### Step 4: Start Small

Your first PR doesn't have to be perfect:
- Fix a typo
- Improve documentation
- Add a test
- Fix a small bug

### Step 5: Learn and Grow

Every expert was once a beginner. We're here to help you grow as a contributor!

## Code of Conduct

This project follows Mozilla's [Community Participation Guidelines](https://www.mozilla.org/about/governance/policies/participation/).

In brief:
- **Be respectful and inclusive**
- **Focus on constructive feedback**
- **Help create a welcoming environment**
- **Report concerns** to maintainers

See our full [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## Questions?

- 💬 Open a [GitHub Discussion](https://github.com/mozilla-ai/any-guardrail/discussions)
- 🐛 Report a [Bug](https://github.com/mozilla-ai/any-guardrail/issues/new?template=bug_report.md)
- 💡 Request a [Feature](https://github.com/mozilla-ai/any-guardrail/issues/new?template=feature_request.md)

We're excited to have you as part of the any-guardrail community! 🚀

---

**License**: By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE) file).
