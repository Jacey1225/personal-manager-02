# Tutorial

This tutorial will guide you through the basic workflow of using the Personal Manager 02 API.

## 1. Sign Up

First, you need to create a new user account. You can do this by sending a `GET` request to the `/auth/signup` endpoint.

**Request:**
```
GET /auth/signup?username=testuser&email=test@example.com&password=password
```

**Response:**
```json
{
  "status": "success",
  "user_id": "your-new-user-id"
}
```
Save the `user_id` from the response, as you will need it for the next steps.

## 2. Log In

Once you have signed up, you can log in to get a session token (although this API does not seem to use session tokens, logging in confirms your credentials).

**Request:**
```
GET /auth/login?username=testuser&password=password
```

**Response:**
```json
{
  "status": "success",
  "user_id": "your-user-id"
}
```

## 3. Create a Project

Now that you are logged in, you can create a new project.

**Request:**
Send a `POST` request to `/projects/create_project` with the following body:
```json
{
  "project_name": "My First Project",
  "project_transparency": true,
  "project_likes": 0,
  "project_members": [["testuser", "admin"]],
  "user_id": "your-user-id"
}
```

**Response:**
```json
{
  "message": "Project created successfully."
}
```

## 4. View Your Project

You can view the project you just created by sending a `GET` request to `/projects/view_project`.

**Request:**
```
GET /projects/view_project?project_id=the-project-id&user_id=your-user-id&project_name=My%20First%20Project
```
**Note:** You will need the `project_id` which is not returned by the create endpoint. This is a potential improvement for the API.

**Response:**
```json
{
  "project": {
    "project_name": "My First Project",
    ...
  },
  "user_data": { ... }
}
```

This concludes the basic tutorial. You can now explore the other API endpoints to manage your projects, tasks, and events.