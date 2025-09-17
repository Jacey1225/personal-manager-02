# Personal Manager - Lazi

A comprehensive AI-powered personal calendar and project management system featuring intelligent natural language processing, multi-user coordination, and cross-platform accessibility.

## 🚀 Overview

Personal Manager (Lazi) is an advanced personal productivity suite that combines machine learning-powered natural language processing with Google Calendar integration to provide intelligent event scheduling, project management, and team coordination capabilities. The system features a FastAPI backend, iOS mobile application, and custom-trained language models for understanding and processing user requests.

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **FastAPI REST API** - High-performance async web framework
- **Custom T5 Language Model** - Fine-tuned for calendar event processing
- **Google Calendar/Tasks Integration** - Real-time sync with Google services
- **Multi-user Authentication** - OAuth 2.0 with Google
- **Project Management System** - Team coordination and availability checking

### Frontend (iOS/SwiftUI)
- **Native iOS Application** - Built with SwiftUI for modern UI/UX
- **Speech Recognition** - Voice-to-text input processing
- **Real-time API Integration** - Seamless backend communication
- **Multi-view Navigation** - Home, Projects, Tasks, Authentication

### Machine Learning Pipeline
- **Custom T5 Model** - Trained on calendar/scheduling data
- **Natural Language Understanding** - Intent recognition and entity extraction
- **Date/Time Processing** - Intelligent datetime parsing and validation
- **Response Generation** - Context-aware response synthesis

## 📁 Project Structure

```
personal-manager-02/
├── api/                          # FastAPI Backend
│   ├── app.py                   # Main application & scheduler endpoints
│   ├── auth_router.py           # Authentication & OAuth routes
│   ├── project_router.py        # Project management endpoints
│   ├── coordination_router.py   # Team coordination & availability
│   └── tasklist_router.py       # Task management routes
├── src/                         # Core Python Modules
│   ├── google_calendar/         # Google API Integration
│   │   ├── enable_google_api.py # OAuth & API setup
│   │   ├── handleEvents.py      # Event CRUD operations
│   │   ├── handleDateTimes.py   # DateTime processing
│   │   └── eventSetup.py        # Calendar service setup
│   ├── model_setup/             # ML Model Components
│   │   ├── TrainingSetup.py     # Model training pipeline
│   │   ├── structureData.py     # Data preprocessing
│   │   ├── structure_model_output.py # Response processing
│   │   └── test_model_accuracy.py    # Model evaluation
│   ├── track_projects/          # Project Management
│   │   ├── handleProjects.py    # Project CRUD & coordination
│   │   └── coordinate_datetimes/# Availability coordination
│   └── validators/              # Data validation & decorators
│       └── validators.py        # Input/output validation
├── ios/Lazi/                    # iOS Application
│   ├── Lazi/
│   │   ├── LaziApp.swift        # App entry point
│   │   ├── ContentView.swift    # Authentication flow
│   │   ├── SidebarView.swift    # Navigation sidebar
│   │   ├── home/                # Chat interface
│   │   ├── projects/            # Project management views
│   │   └── tasks/               # Task list views
│   └── Lazi.xcodeproj/          # Xcode project configuration
├── model/                       # Trained ML Models
│   ├── model.safetensors        # Production model weights
│   ├── config.json             # Model configuration
│   └── checkpoint-*/           # Training checkpoints
├── data/                        # Application Data
│   ├── credentials.json         # Google API credentials
│   ├── event_scheduling.csv     # Training data
│   ├── processed_event_data.pt  # Preprocessed training data
│   ├── tokens/                  # OAuth tokens
│   └── users/                   # User profiles & projects
└── tests/                       # Test Suite
    └── test_route_results.py    # API endpoint tests
```

## 🔧 Core Features

### 1. Intelligent Natural Language Processing
- **Custom T5 Model**: Fine-tuned transformer model for calendar event understanding
- **Intent Recognition**: Automatically classifies user requests (add, delete, update events)
- **Entity Extraction**: Identifies event names, dates, times, and participants
- **Context Awareness**: Understands relative dates ("tomorrow", "next week")

### 2. Google Calendar Integration
- **Real-time Sync**: Bidirectional synchronization with Google Calendar
- **Event Management**: Create, read, update, delete calendar events
- **Task Integration**: Google Tasks support for todo items
- **Multi-user Access**: OAuth authentication for multiple users

### 3. Project Management & Team Coordination
- **Project Creation**: Create projects with team members
- **Availability Checking**: Check team member availability across time ranges
- **Event Coordination**: Schedule project events with automatic conflict detection
- **Member Management**: Add/remove project members dynamically

### 4. iOS Mobile Application
- **Speech Recognition**: Voice input for natural language requests
- **Real-time Chat**: Conversational interface for event management
- **Project Views**: Visual project management with member coordination
- **Authentication**: Secure Google OAuth integration

## 🛠️ Technical Stack

### Backend Technologies
- **FastAPI** - Modern async web framework
- **Transformers** - Hugging Face transformers library
- **PyTorch** - Deep learning framework
- **Google APIs** - Calendar, Tasks, OAuth
- **Pydantic** - Data validation and serialization
- **APScheduler** - Background task scheduling

### Frontend Technologies
- **SwiftUI** - Modern iOS UI framework
- **URLSession** - HTTP networking
- **Speech Framework** - Voice recognition
- **Foundation** - Core iOS utilities

### Machine Learning
- **T5 (Text-to-Text Transfer Transformer)** - Base model architecture
- **Custom Training Pipeline** - Fine-tuned on calendar/scheduling data
- **spaCy** - Natural language processing utilities
- **NumPy/Pandas** - Data manipulation and analysis

## 📋 API Endpoints

### Scheduler Routes (`/scheduler`)
- `POST /fetch_events` - Process natural language input and extract events
- `POST /process_input` - Execute calendar operations (add/delete/update)
- `POST /delete_event/{event_id}` - Delete specific calendar events

### Authentication Routes (`/auth`)
- `POST /signup` - User registration
- `POST /login` - User authentication
- `GET /google-auth` - Initiate Google OAuth flow
- `POST /google-auth/complete` - Complete OAuth flow

### Project Management Routes (`/projects`)
- `POST /create_project` - Create new projects with team members
- `GET /list` - List user's projects
- `GET /events/{project_id}` - Get project-specific events
- `GET /add_member` - Add members to projects
- `GET /delete_member` - Remove members from projects

### Coordination Routes (`/coordinate`)
- `GET /fetch-users` - Retrieve user information by email
- `POST /get_availability` - Check availability across team members

### Task Routes (`/task_list`)
- `POST /list_events` - Retrieve user's calendar events

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- iOS 14+ (for mobile app)
- Xcode 12+ (for iOS development)
- Google Cloud Project with Calendar/Tasks APIs enabled

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd personal-manager-02
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Google API credentials**
```bash
# Place your Google API credentials in data/credentials.json
# Enable Google Calendar API and Google Tasks API in Google Cloud Console
```

4. **Start the FastAPI server**
```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

5. **Optional: Setup ngrok for external access**
```bash
ngrok http 8000
```

### iOS Setup

1. **Open the Xcode project**
```bash
cd ios/Lazi
open Lazi.xcodeproj
```

2. **Update API endpoints**
- Update the base URL in all Swift files to point to your backend
- Replace `https://29098e308ec4.ngrok-free.app` with your server URL

3. **Build and run**
- Select your target device or simulator
- Build and run the project (⌘+R)

## 🔄 Data Flow

### 1. User Input Processing
```
User Voice/Text Input → Speech Recognition → FastAPI Backend → T5 Model → Event Extraction
```

### 2. Calendar Operations
```
Processed Events → Google Calendar API → Calendar Updates → Response Generation → iOS Display
```

### 3. Project Coordination
```
Project Creation → Member Addition → Availability Checking → Event Scheduling → Notification
```

## 🧠 Machine Learning Components

### Model Training Pipeline
- **Data Collection**: Calendar event data in CSV format
- **Preprocessing**: Text normalization and tokenization
- **Training**: Fine-tuning T5 model on scheduling tasks
- **Evaluation**: Accuracy testing and validation
- **Deployment**: Model serialization and API integration

### Model Architecture
- **Base Model**: T5-small (60M parameters)
- **Custom Fine-tuning**: Trained on calendar/scheduling domain
- **Input Format**: Natural language text
- **Output Format**: Structured event data (JSON)

## 🔐 Security & Authentication

### OAuth 2.0 Integration
- Google OAuth for user authentication
- Secure token storage and refresh
- Multi-user support with isolated data

### API Security
- CORS middleware for cross-origin requests
- Request validation with Pydantic models
- Error handling and logging

## 📊 Project Insights

### Codebase Statistics
- **Total Lines**: ~2,000+ lines of code
- **Languages**: Python (Backend), Swift (iOS), JSON (Config)
- **Files**: 50+ source files across backend and frontend
- **APIs**: 15+ REST endpoints

### Key Components
- **Custom ML Model**: Fine-tuned T5 transformer
- **Multi-platform**: iOS app + Python backend
- **Real-time Sync**: Google Calendar integration
- **Team Coordination**: Multi-user project management

## 🚧 Development Status

### Completed Features
- ✅ Natural language event processing
- ✅ Google Calendar integration
- ✅ iOS application with voice input
- ✅ Project management system
- ✅ Multi-user authentication
- ✅ Team availability coordination

### Future Enhancements
- 🔄 Advanced ML model improvements
- 🔄 Web application interface
- 🔄 Push notifications
- 🔄 Calendar analytics and insights
- 🔄 Integration with other calendar services

## 📝 Contributing

This is a personal project showcasing AI-powered calendar management and cross-platform development. The codebase demonstrates:

- **Machine Learning Integration**: Custom T5 model training and deployment
- **API Development**: FastAPI backend with comprehensive endpoints
- **Mobile Development**: Native iOS app with SwiftUI
- **System Integration**: Google API integration and OAuth
- **Project Architecture**: Modular, scalable design patterns

## 📄 License

This project is developed for educational and portfolio purposes, demonstrating modern software development practices, machine learning integration, and cross-platform application development.

---

**Personal Manager (Lazi)** - Intelligent calendar management powered by AI and designed for modern productivity workflows.