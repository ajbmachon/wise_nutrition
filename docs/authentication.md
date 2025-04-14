# Firebase Authentication Integration

This document explains how Firebase Authentication is integrated into the Wise Nutrition API.

## Setup

Firebase Authentication is used for user registration, authentication, and management. The integration uses the Firebase Admin SDK for server-side operations.

### Prerequisites

- A Firebase project with Authentication enabled
- Firebase credentials file (`firebase_credentials.json`)
- Firebase Admin SDK (`firebase-admin` package)

### Configuration

The Firebase Admin SDK is initialized in `wise_nutrition/auth/firebase_auth.py` using credentials from `firebase_credentials.json`.

```python
cred_path = os.path.join(os.getcwd(), "firebase_credentials.json")
if os.path.exists(cred_path):
    cred = credentials.Certificate(cred_path)
    firebase_app = firebase_admin.initialize_app(cred)
```

## Authentication Flow

### User Registration

1. User submits registration data (email, password, etc.)
2. Server creates the user in Firebase Authentication
3. Server returns user information (without sensitive data)

### User Login

1. User submits email and password
2. Server verifies credentials with Firebase
3. Server generates a custom token for the user
4. Client uses this token for subsequent authenticated requests

### Token Authentication

1. Client includes token in Authorization header (`Bearer <token>`)
2. Server verifies the token using Firebase Admin SDK
3. Server identifies the user and processes the request

## API Endpoints

### Authentication Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/register` | POST | Register a new user |
| `/api/v1/auth/login` | POST | Authenticate and get token |
| `/api/v1/auth/me` | GET | Get current user information |
| `/api/v1/auth/me` | PUT | Update current user information |
| `/api/v1/auth/password-reset` | POST | Request password reset email |
| `/api/v1/auth/verify-email` | POST | Verify user email with code |

### Restricted Endpoints

These endpoints require authentication:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/nutrition_rag_chain/invoke` | POST | Get nutrition advice (authenticated) |

### Public Endpoints

These endpoints are available without authentication:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/nutrition_rag_chain/public` | POST | Get nutrition advice (public) |
| `/api/v1/health` | GET | Health check |

## Models

User-related models are defined in `wise_nutrition/models/user.py`:

- `UserBase`: Common user fields
- `UserCreate`: For user registration
- `UserLogin`: For user login
- `UserResponse`: For user data responses
- `Token`: For authentication tokens

## Security Notes

1. Passwords are never stored in our database; Firebase handles password hashing and storage
2. JWT tokens are used for authentication
3. HTTPS should be used in production to secure data transmission
4. Sensitive operations (password reset, user deletion) have additional security checks

## Client Usage Example

### Registration

```javascript
// Example JavaScript client code
async function registerUser(userData) {
  const response = await fetch('/api/v1/auth/register', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(userData),
  });
  return response.json();
}
```

### Authentication

```javascript
// Example JavaScript client code
async function loginUser(credentials) {
  const response = await fetch('/api/v1/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(credentials),
  });
  const data = await response.json();
  // Store token for later use
  localStorage.setItem('token', data.access_token);
  return data;
}
```

### Authenticated Request

```javascript
// Example JavaScript client code
async function getNutritionAdvice(query) {
  const token = localStorage.getItem('token');
  const response = await fetch('/api/v1/nutrition_rag_chain/invoke', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify({ query }),
  });
  return response.json();
}
``` 