# Widget API Test Suite

Comprehensive integration tests for the Widget API OAuth2 flow and public endpoints.

## Test Overview

This test suite validates the complete widget creation and interaction workflow:

1. **OAuth2 Authentication Flow** - Tests token acquisition and user verification
2. **Widget Creation via SDK** - Tests the WriteWidget SDK for creating widgets programmatically
3. **Public API Interactions** - Tests widget endpoint execution through the public API
4. **Error Handling** - Validates proper error responses for edge cases
5. **Data Persistence** - Confirms test data remains in database for manual review

## Features

### Comprehensive Logging
- **Dual output**: Console (INFO) and file (DEBUG)
- **Structured test sections**: Clear separators between test phases
- **Detailed operation logging**: Every API call, database operation, and validation step is logged
- **Token payload inspection**: JWT token contents are decoded and displayed
- **Database verification**: Confirms data persistence at each step

### Test Coverage

#### Test 01: OAuth2 Token Acquisition
- Validates password grant flow
- Tests token structure and expiration
- Decodes and displays JWT payload
- Verifies token type and format

#### Test 02: OAuth2 User Verification
- Tests user verification endpoint
- Validates developer privileges
- Confirms token payload accuracy
- Checks user details and scopes

#### Test 03: Widget Creation via SDK
- Initializes WriteWidget SDK with OAuth2 credentials
- Creates a counter widget with:
  - Name and size configuration
  - Interaction endpoint (`/increment`)
  - Text display component
  - Button component
- Saves widget to database
- Associates widget with project
- Verifies database persistence

#### Test 04: Public Widget Interaction
- Executes widget logic through public API
- Tests single interaction
- Tests multiple sequential interactions
- Validates response codes and data

#### Test 05: Public API Error Handling
- Tests non-existent widget ID
- Tests widget not associated with project
- Validates proper HTTP error codes (404)
- Confirms error message accuracy

#### Test 06: Data Persistence Verification
- Confirms user data persisted
- Confirms project data persisted
- Confirms widget data persisted
- Provides cleanup instructions

## Running the Tests

### Prerequisites

```bash
# Ensure all dependencies are installed
pip install -r requirements/api.txt
```

### Environment Setup

Ensure you have a `.env` file with:
```env
JWT_SECRET=your_jwt_secret_key_here
```

### MongoDB Connection

The tests use the following MongoDB collections:
- `userAuthDatabase.userCredentials` - User accounts
- `userAuthDatabase.openProjects` - Projects
- `userAuthDatabase.openWidgets` - Widget configurations

### Execute Tests

#### Run all tests with verbose output:
```bash
pytest api/tests/test_widget_api_flow.py -v -s
```

#### Run with live logging:
```bash
pytest api/tests/test_widget_api_flow.py -v -s --log-cli-level=INFO
```

#### Run specific test class:
```bash
pytest api/tests/test_widget_api_flow.py::TestOAuth2Flow -v -s
pytest api/tests/test_widget_api_flow.py::TestWidgetCreation -v -s
pytest api/tests/test_widget_api_flow.py::TestPublicAPIEndpoints -v -s
```

#### Run specific test:
```bash
pytest api/tests/test_widget_api_flow.py::TestOAuth2Flow::test_01_oauth_token_acquisition -v -s
```

#### Run from Python directly:
```bash
python -m api.tests.test_widget_api_flow
```

## Test Output

### Console Output
The tests produce detailed console output with:
- Clear section separators (`====`)
- Step-by-step operation logs
- Success indicators (✓)
- Response data and payloads
- Database verification results

### Log File
All test execution details are saved to:
- `test_widget_api_flow.log` - Detailed test execution log
- `tests/test_widget_api.log` - Additional pytest logs (if pytest.ini is used)

## Test Data

### Data Preservation
**IMPORTANT**: Test data is **NOT** automatically cleaned up after test execution. This allows for manual verification and inspection.

### Generated Test Data

Each test run creates:

1. **Test User**
   - Username: `test_widget_user_<random>`
   - Email: `test_<random>@example.com`
   - Password: `TestPassword123!`
   - Developer status: `true`

2. **Test Project**
   - Name: `Test Widget Project`
   - Project ID: UUID
   - Associated widgets array

3. **Test Widget**
   - Name: `Test Counter Widget`
   - Size: `medium`
   - Interaction endpoint: `/increment`
   - Components: text display + button

### Manual Cleanup

The test completion log provides the exact IDs for cleanup:

```bash
# Connect to MongoDB
mongo

# Switch to database
use userAuthDatabase

# Remove test user
db.userCredentials.deleteOne({"user_id": "USER_ID_FROM_LOG"})

# Remove test project
db.openProjects.deleteOne({"project_id": "PROJECT_ID_FROM_LOG"})

# Remove test widget
db.openWidgets.deleteOne({"id": "WIDGET_ID_FROM_LOG"})
```

Or delete all test data by username:
```javascript
db.userCredentials.deleteOne({"username": /^test_widget_user_/})
```

## Troubleshooting

### Common Issues

#### 1. JWT_SECRET not found
```
Error: No secret key found
```
**Solution**: Add `JWT_SECRET` to your `.env` file

#### 2. MongoDB connection failed
```
Error: Connection refused
```
**Solution**: Ensure MongoDB is running and accessible

#### 3. Import errors
```
ImportError: No module named 'api'
```
**Solution**: Run tests from project root or add project to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/personal-manager-02"
```

#### 4. Async test errors
```
RuntimeError: Event loop is closed
```
**Solution**: Ensure `pytest-asyncio` is installed and `pytest.ini` is configured

### Debug Mode

For maximum verbosity:
```bash
pytest api/tests/test_widget_api_flow.py -vv -s --log-cli-level=DEBUG --tb=long
```

## Test Architecture

### Fixtures

- `async_client`: AsyncClient for FastAPI testing
- `setup_test_user`: Creates test user with developer privileges and project

### Test Context

The `test_context` dictionary maintains state across tests:
- `access_token`: OAuth2 access token
- `user_id`: Created user ID
- `widget_id`: Created widget ID
- `project_id`: Created project ID

### Test Order

Tests are numbered and should run in order:
1. OAuth2 flow must complete before widget creation
2. Widget creation must complete before public API tests
3. All tests must complete before data persistence verification

## Integration with CI/CD

To integrate with CI/CD pipelines, you may want to add cleanup:

```python
@pytest.fixture(scope="module", autouse=True)
async def cleanup_after_tests():
    yield
    # Add cleanup logic here if needed for CI/CD
    if os.getenv("CI"):
        # Clean up test data
        pass
```

## Contributing

When adding new tests:
1. Follow the numbering convention (test_01, test_02, etc.)
2. Add comprehensive logging at each step
3. Use clear section separators
4. Document expected outcomes
5. Update this README with new test descriptions

## Support

For issues or questions:
1. Check the log files for detailed error information
2. Verify all prerequisites are installed
3. Ensure MongoDB is running and accessible
4. Confirm environment variables are set correctly
