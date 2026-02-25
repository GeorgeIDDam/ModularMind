# Contributing to JIDM_v2

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a new branch for your feature: `git checkout -b feature-name`

## Development Workflow
1. Make your changes
2. Test your changes thoroughly
3. Commit with clear, descriptive messages
4. Push to your branch
5. Create a pull request

## Code Style
- Follow PEP 8 guidelines for Python code
- Use type hints where applicable
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Testing
- Test your changes before committing
- Ensure no regressions in existing functionality

## Commit Message Format
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat: add new attention mechanism to model

Implemented multi-head attention with configurable heads.
Updated model config to support new parameters.

Closes #123
```

## Questions?
Contact the project maintainer for any questions or clarifications.
