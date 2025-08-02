# Deployment and Release Setup

## GitHub Repository Secrets

To enable CI/CD and PyPI publishing, add these secrets to your GitHub repository:

### Required Secrets

1. **PYPI_API_TOKEN**
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token with scope "Entire account"
   - Add as repository secret

2. **TEST_PYPI_API_TOKEN** (optional but recommended)
   - Go to https://test.pypi.org/manage/account/token/
   - Create a new API token with scope "Entire account"
   - Add as repository secret

### Setting up secrets:

1. Go to your GitHub repository
2. Click Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add each secret with the exact names above

## Release Process

### Automatic Release (Recommended)

1. Update version in `pyproject.toml`
2. Commit and push changes
3. Create and push a git tag:
   ```bash
   git tag v0.0.1
   git push origin v0.0.1
   ```
4. GitHub Actions will automatically:
   - Run tests
   - Build package
   - Publish to Test PyPI (if token provided)
   - Publish to PyPI
   - Create GitHub release

### Manual Release

```bash
# Build package
python -m build

# Check package
twine check dist/*

# Upload to Test PyPI (optional)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## CI/CD Workflows

### CI Workflow (`.github/workflows/ci.yml`)
- Runs on push to main/develop and PRs
- Tests on Python 3.9-3.12
- Linting with ruff and black
- Type checking with mypy
- Unit tests with coverage
- Integration tests with RabbitMQ
- Security scanning with bandit and safety

### Release Workflow (`.github/workflows/release.yml`)
- Triggers on git tags (v*)
- Builds and validates package
- Publishes to Test PyPI first
- Publishes to PyPI
- Creates GitHub release

## Environment Setup

### Protection Rules (Recommended)

1. Go to Settings → Environments
2. Create "release" environment
3. Add protection rules:
   - Required reviewers
   - Deployment branches: only tags matching `v*`

### Branch Protection

1. Go to Settings → Branches
2. Add rule for `main` branch:
   - Require status checks (CI workflow)
   - Require up-to-date branches
   - Require linear history (optional)

## First Release Checklist

- [ ] Update author email in `pyproject.toml`
- [ ] Update repository URLs in `pyproject.toml`
- [ ] Add PyPI API tokens to GitHub secrets
- [ ] Test the package build: `python -m build`
- [ ] Create initial git tag: `git tag v0.0.1`
- [ ] Push tag: `git push origin v0.0.1`
- [ ] Verify release workflow completes successfully
- [ ] Check package appears on PyPI: https://pypi.org/project/llmq/