# React + FastAPI Full Stack Application

This is a full-stack application with a React frontend and FastAPI backend.

## Project Structure

```
.
├── frontend/          # React frontend
└── backend/          # FastAPI backend
```

## Setup Instructions

### Quick Start

1. Install all dependencies:
   ```bash
   npm run install-all
   ```

2. Start both frontend and backend servers:
   ```bash
   npm start
   ```

This will start:
- Frontend at `http://localhost:5173`
- Backend at `http://localhost:8000`

### Detailed Setup

#### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Features

- Modern React frontend with TypeScript
- FastAPI backend with CORS support
- Tailwind CSS for styling
- React Query for data fetching
- Axios for HTTP requests
- Concurrent development server management

