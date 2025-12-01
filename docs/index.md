# Welcome to Personal Manager 02

Personal Manager 02 is a powerful project management platform with a comprehensive **public SDK** and **OAuth2-secured API** that enables developers to build custom widgets, automation, and integrations.

## What is Personal Manager 02?

Personal Manager 02 is a comprehensive solution for managing your tasks, events, projects, and more. It provides a flexible API that allows you to integrate it with your existing workflows and build custom applications on top of it.

The project is built with a modern tech stack, including FastAPI for the backend API, and is designed to be extensible and easy to use.

## Key Features

### 🔐 **OAuth2 Authentication**
- Secure token-based authentication with refresh tokens
- Granular permission scopes (read, write, delete, admin)
- Access tokens (1 hour) and refresh tokens (30 days)
- API key support for developers

### 📦 **Public Python SDK**
- Simple, Pythonic interface for widget management
- Automatic authentication handling
- Type hints and IDE support
- Install with: `pip install Lazi-sdk`

### 🎨 **Custom Widgets**
- Create custom widgets with your own logic
- Dynamic endpoint registration
- Attach media files (images, videos, documents)
- Real-time data refresh intervals
- User-defined parameters and headers

### 🔧 **Extensibility**
- Create custom API endpoints for widgets
- Execute user-defined Python code securely
- Build automation workflows
- Integrate with external APIs

### 📊 **Project Management**
- Organize work into projects
- Manage team members and permissions
- Track progress and metrics
- Collaborate through discussions

### ☁️ **Cloud Storage**
- AWS S3 integration for media files
- Presigned URLs for secure access
- Support for images, videos, PDFs, and more

## Quick Start

### Installation

```bash
pip install Lazi-sdk
```

### Basic Usage

```python
from lazi_sdk import WriteWidget

# Initialize with OAuth token
widget = WriteWidget(
    token="your_oauth_token",
    user_id="user_12345"
)

# Create a widget
await widget.create_new(
    name="Analytics Dashboard",
    description="Real-time sales metrics",
    size="large",
    project_id="project_001"
)

# Save to project
await widget.save()
```

## Documentation Sections

- **[Getting Started](tutorial.md)** - Step-by-step tutorial for new users
- **[SDK Guide](sdk/installation.md)** - Complete Python SDK documentation
- **[API Reference](api.md)** - Detailed API endpoint documentation
- **[Advanced Topics](advanced/scopes.md)** - OAuth2 scopes, custom endpoints, security

## Authentication Flow

1. **Register** your application to get API credentials
2. **Authenticate** users with OAuth2 to get access tokens
3. **Make API calls** using the access token
4. **Refresh tokens** automatically when they expire (30 days)

## Use Cases

- 📈 **Build custom dashboards** with real-time data
- 🤖 **Create automation workflows** for project management
- 🔗 **Integrate external services** (GitHub, Slack, analytics tools)
- 📱 **Develop mobile apps** with widget support
- 🎯 **Custom business logic** executed on your infrastructure

## Getting Started

To get started with the Personal Manager 02 API, check out the [Tutorial](tutorial.md) for a step-by-step guide on how to perform basic operations.

For SDK usage, see the [SDK Installation Guide](sdk/installation.md).

For a detailed description of all the available API endpoints, please refer to the [API Reference](api.md).

## Contributing

This project is open source and we welcome contributions from the community. If you would like to contribute, please check out the [GitHub repository](https://github.com/Jacey1225/personal-manager-02).

## Support

- 📖 [Documentation](https://jacey1225.github.io/personal-manager-02/)
- 💬 [GitHub Discussions](https://github.com/Jacey1225/personal-manager-02/discussions)
- 🐛 [Issue Tracker](https://github.com/Jacey1225/personal-manager-02/issues)
