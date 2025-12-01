# SDK Installation Guide

## Installation

Install the Lazi SDK using pip:

```bash
pip install Lazi-sdk
```

### Requirements

- Python 3.9 or higher
- Active internet connection for API calls

### Dependencies

The SDK automatically installs the following dependencies:

- `fastapi>=0.104.0` - Web framework support
- `pydantic>=2.0.0` - Data validation
- `motor>=3.3.0` - Async MongoDB driver
- `pymongo>=4.5.0` - MongoDB support
- `python-dotenv>=1.0.0` - Environment variables
- `certifi>=2023.7.22` - SSL certificates
- `boto3>=1.28.0` - AWS S3 integration
- `botocore>=1.31.0` - AWS SDK core
- `passlib[bcrypt]>=1.7.4` - Password hashing
- `pyjwt>=2.8.0` - JWT token handling
- `python-multipart>=0.0.6` - File uploads

## Development Installation

For contributing to the SDK:

```bash
# Clone the repository
git clone https://github.com/Jacey1225/personal-manager-02.git
cd personal-manager-02

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Development Dependencies

```bash
pip install Lazi-sdk[dev]
```

Includes:
- `pytest` - Testing framework
- `black` - Code formatter
- `ruff` - Linter
- `mypy` - Type checker
- `pre-commit` - Git hooks

## Environment Setup

Create a `.env` file in your project root:

```env
# API Configuration
API_BASE_URL=https://api.lazi.com
JWT_SECRET=your-secret-key

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017

# AWS S3 Configuration (optional, for media uploads)
AWS_ACCESS_KEY=your-access-key
AWS_SECRET_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
```

## Verify Installation

```python
import lazi_sdk

print(lazi_sdk.__version__)
# Output: 0.1.0
```

## Quick Start

After installation, see the [Authentication Guide](authentication.md) to get started with the SDK.

## Troubleshooting

### Import Errors

If you encounter import errors:

```bash
# Upgrade pip
pip install --upgrade pip

# Reinstall the package
pip uninstall Lazi-sdk
pip install Lazi-sdk
```

### SSL Certificate Issues

If you encounter SSL errors:

```bash
pip install --upgrade certifi
```

### Async/Await Issues

Make sure you're using Python 3.9+ and running async functions with:

```python
import asyncio

async def main():
    # Your async code here
    pass

asyncio.run(main())
```

## Next Steps

- [Authentication Guide](authentication.md) - Learn how to authenticate with OAuth2
- [Widget Management](widgets.md) - Create and manage widgets
- [Examples](examples.md) - See complete code examples
