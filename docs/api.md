# API Reference

This API documentation provides details on the available endpoints for the Personal Manager 02 application.

## Authentication

The authentication endpoints handle user signup, login, and other authentication-related operations.

### Signup

*   **Endpoint:** `GET /auth/signup`
*   **Method:** `GET`
*   **Description:** Signs up a new user.
*   **Query Parameters:**
    *   `username` (str, required): The username for the new user.
    *   `email` (str, required): The email address for the new user.
    *   `password` (str, required): The password for the new user.
    *   `project_id` (str, optional): A project ID to associate with the user.
    *   `org_id` (str, optional): An organization ID to associate with the user.
*   **Sample Request:**
    ```
    GET /auth/signup?username=testuser&email=test@example.com&password=password
    ```
*   **Success Response:**
    ```json
    {
      "status": "success",
      "user_id": "some-uuid"
    }
    ```

### Login

*   **Endpoint:** `GET /auth/login`
*   **Method:** `GET`
*   **Description:** Logs in an existing user.
*   **Query Parameters:**
    *   `username` (str, required): The username of the user.
    *   `password` (str, required): The password of the user.
*   **Sample Request:**
    ```
    GET /auth/login?username=testuser&password=password
    ```
*   **Success Response:**
    ```json
    {
      "status": "success",
      "user_id": "some-uuid"
    }
    ```

### Set iCloud User

*   **Endpoint:** `POST /auth/set_icloud_user`
*   **Method:** `POST`
*   **Description:** Sets the iCloud credentials for a user.
*   **Request Body:**
    ```json
    {
      "service_name": "user_auth",
      "apple_user": "apple_username",
      "apple_pass": "apple_password"
    }
    ```

### Remove User

*   **Endpoint:** `POST /auth/remove_user`
*   **Method:** `POST`
*   **Description:** Removes a user from the system.
*   **Request Body:**
    ```json
    {
      "user_id": "some-uuid"
    }
    ```

### Google Authentication

*   **Endpoint:** `GET /auth/google`
*   **Method:** `GET`
*   **Description:** Initiates the Google OAuth2 flow and returns an authorization URL.
*   **Query Parameters:**
    *   `user_id` (str, required): The ID of the user.

### Complete Google Authentication

*   **Endpoint:** `POST /auth/google/complete`
*   **Method:** `POST`
*   **Description:** Completes the Google OAuth2 flow with the authorization code.
*   **Request Body:**
    ```json
    {
      "user_id": "some-uuid",
      "authorization_code": "some-auth-code"
    }
    ```

## Coordination

Endpoints for coordinating between users.

### Fetch Users

*   **Endpoint:** `POST /coordinate/fetch_users`
*   **Method:** `POST`
*   **Description:** Fetches user data.
*   **Request Body:**
    ```json
    {
      "members": [
        {"email": "user1@example.com"},
        {"email": "user2@example.com"}
      ]
    }
    ```

### Get Availability

*   **Endpoint:** `POST /coordinate/get_availability`
*   **Method:** `POST`
*   **Description:** Gets the availability of users for a specific time range.
*   **Request Body:**
    ```json
    {
      "users": [{"user_id": "user1", "username": "testuser"}],
      "request_start": "2025-01-01T10:00:00",
      "request_end": "2025-01-01T11:00:00"
    }
    ```

## Discussions

Endpoints for managing discussions within projects.

**Note:** Some of these endpoints have incorrect parameter definitions (multiple body parameters). The documentation assumes they will be refactored to have only one body parameter, with other parameters passed as query parameters.

### View Discussion

*   **Endpoint:** `GET /discussions/view_discussion`
*   **Method:** `GET`
*   **Description:** Fetches an existing discussion.
*   **Query Parameters:**
    *   `user_id` (str, required)
    *   `project_id` (str, required)
    *   `discussion_id` (str, required)
    *   `force_refresh` (bool, optional)

### List Project Discussions

*   **Endpoint:** `GET /discussions/list_project_discussions`
*   **Method:** `GET`
*   **Description:** Lists all discussions for a project.
*   **Query Parameters:**
    *   `user_id` (str, required)
    *   `project_id` (str, required)
    *   `force_refresh` (bool, optional)

## Events

Endpoints for managing events and tasks.

### Process Input

*   **Endpoint:** `POST /scheduler/process_input`
*   **Method:** `POST`
*   **Description:** Processes natural language input to create, update, or delete events.
*   **Request Body:**
    ```json
    {
      "input_text": "Schedule a meeting with John tomorrow at 2pm",
      "user_id": "some-uuid"
    }
    ```

### Delete Event

*   **Endpoint:** `POST /scheduler/delete_event/{event_id}`
*   **Method:** `POST`
*   **Description:** Deletes an event.
*   **Path Parameters:**
    *   `event_id` (str, required)
*   **Request Body:**
    ```json
    {
      "user_id": "some-uuid",
      "event_requested": { ... }
    }
    ```

### Update Event

*   **Endpoint:** `POST /scheduler/update_event/{event_id}`
*   **Method:** `POST`
*   **Description:** Updates an event.
*   **Path Parameters:**
    *   `event_id` (str, required)
*   **Request Body:**
    ```json
    {
      "user_id": "some-uuid",
      "event_requested": { ... }
    }
    ```

## Organizations

Endpoints for managing organizations.

**Note:** Some of these endpoints have incorrect parameter definitions. The documentation assumes they will be refactored.

### Create Organization

*   **Endpoint:** `POST /organizations/create_org`
*   **Method:** `POST`
*   **Description:** Creates a new organization.
*   **Query Parameters:**
    *   `user_id` (str, required)
*   **Request Body:**
    ```json
    {
      "name": "My Organization",
      "members": ["user1", "user2"],
      "projects": ["project1"]
    }
    ```

## Projects

Endpoints for managing projects.

**Note:** Some of these endpoints have incorrect parameter definitions. The documentation assumes they will be refactored.

### View Project

*   **Endpoint:** `GET /projects/view_project`
*   **Method:** `GET`
*   **Description:** Fetches an existing project.
*   **Query Parameters:**
    *   `project_id` (str, required)
    *   `user_id` (str, required)
    *   `project_name` (str, required)
    *   `force_refresh` (bool, optional)

## Task List

Endpoints for managing task lists.

### List Events

*   **Endpoint:** `POST /task_list/list_events`
*   **Method:** `POST`
*   **Description:** Lists events for a user.
*   **Request Body:**
    ```json
    {
      "calendar_event": { ... },
      "user_id": "some-uuid",
      "minTime": "2025-01-01T00:00:00Z",
      "maxTime": "2025-01-31T23:59:59Z"
    }
    ```