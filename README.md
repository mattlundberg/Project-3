# React + FastAPI Full Stack Application

This is a full-stack application with a React frontend and FastAPI backend.

## Project Structure

```
.
├── frontend/          # React frontend
└── backend/          # FastAPI backend
```

## Setup Instructions

### Backend Setup

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

4. Run the backend server:
   ```bash
   uvicorn main:app --reload
   ```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:5173`

## Features

- Modern React frontend with TypeScript
- FastAPI backend with CORS support
- Tailwind CSS for styling
- React Query for data fetching
- Axios for HTTP requests

