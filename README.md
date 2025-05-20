# Project-3: The Onionator: Peeling Back Fake News One Headline at a Time
## Slack Channel
Join our Slack channel for project discussions and updates:
- Channel: #404-not-found
- Link: [404 Not Found](https://aiptwestnovem-cki2893.slack.com/archives/C089LSTUQER)

## Team Members
- Tiffany Jimenez
- Sam Lara
- Matthew Lundberg
- Jason Smoody
- Erin Spencer-Priebe 
 
 ## Project Milestones

| Milestone | Due Date | Status |
|----------|----------|----------|
| Project Ideation | 4/29/25 | Complete |
| [Back End] Data Fetching & Clean Up | 5/5/25 | Complete |
| [Back End] Model Build | 5/12/25 | Complete |
| [Front End] Build Title Page | 5/8/25 | Complete |
| [Front End] Build Main/App Page | 5/8/25 | Complete |
| [Front End] Refine Main/App Features | 5/12/25 | Complete |
| Bridge Front and Back End | 5/13/25 | Complete |
| Create Presentation | 5/15/25 | Complete |

## Proposal
Did you hear about the latest headline claiming a celebrity turned into an avocado? No? Well, that's exactly the kind of fake news-and viral satire-that floods our social media feeds every day. These outrageous posts are as wild as your uncle's conspiracy theories after one too many holiday dinners. Scrolling through your feed can feel like diving into a maze of confusing and sometimes hilarious stories that make you question reality (and your family gatherings). 

Our proposal is to build a model that takes in everything from suspicious headlines to eyebrow-raising tweets and tells you whether you're about to share real news, clickbait, or the latest viral joke-saving you from spreading "BREAKING: Celebrity Turns Into Avocado" level nonsense.

## Slide Deck
[The Onionator: Peeling Back Fake News One Headline at a Time](TBD)

## Data sets
Liar Data Set: https://paperswithcode.com/dataset/liar

## Future Research Questions and Issues
- How can we improve model accuracy and reduce false positives/negatives?
- How can we handle evolving misinformation tactics?
- What are the ethical considerations?
- How can we scale the application?
- What additional features would enhance user experience?


## Installation Guide

#### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for version control)

#### Method 1: Using requirements.txt (Recommended)
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd Project-3
   ```

2. Create a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```

#### Method 2: Manual Installation
If you prefer to install packages individually, you can use pip:

```bash
# Core Data Science
pip install pandas==2.1.4 numpy==1.26.2 scipy==1.11.4

# Machine Learning
pip install tensorflow==2.15.0 torch==2.1.2 scikit-learn==1.3.2
pip install transformers==4.36.2 sentence-transformers

# Data Visualization
pip install matplotlib==3.8.2 seaborn==0.13.0 plotly==5.18.0

# Development Tools
pip install jupyter==1.0.0 ipykernel==6.28.0 notebook==7.0.6
pip install black==23.12.1 pylint==3.0.3 pytest==7.4.3

# Web Framework
pip install reflex==0.7.10
```

#### Troubleshooting Common Issues

1. **CUDA/GPU Support**
   - For TensorFlow GPU support:
     ```bash
     pip install tensorflow-gpu
     ```
   - For PyTorch GPU support, visit: https://pytorch.org/get-started/locally/

2. **Version Conflicts**
   - If you encounter version conflicts, try:
     ```bash
     pip install --upgrade pip
     pip install -r src/requirements.txt --no-deps
     ```

3. **Memory Issues**
   - If you encounter memory issues during installation:
     ```bash
     pip install --no-cache-dir -r src/requirements.txt
     ```

4. **Virtual Environment Issues**
   - If you have trouble activating the virtual environment:
     ```bash
     # Windows
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     
     # macOS/Linux
     chmod +x venv/bin/activate
     ```


## Running app
- Navigate to the 'src' folder 
```bash
cd src
```
-Run reflex command
```bash
reflex run
```

# Program Information

## Overview
The Onionator is a web application that uses machine learning to classify news headlines and text as either real news or potential misinformation/satire. The app provides an intuitive interface for users to input text and receive instant feedback on its credibility.

### Core Features
- Text input field for news headlines/content
- Real-time classification using trained ML model
- Visual feedback with custom graphics
- Engaging loading animations
- Light/dark mode toggle
- Mobile-responsive design

### Programming Languages
- Python (Backend/ML)
- JavaScript/TypeScript (Frontend)
- HTML/CSS
- SQL (Database)

### Required Libraries/Dependencies

#### Core Data Science and Numerical Computing
- Pandas - Data processing and analysis
- NumPy - Numerical operations
- SciPy - Scientific computing

#### Machine Learning and Deep Learning
- TensorFlow/Keras - ML model development
- PyTorch - Deep learning framework
- Scikit-learn - ML preprocessing and traditional algorithms
- Transformers - NLP models and text processing
- SentenceTransformers - Text embedding generation

#### Data Visualization
- Matplotlib - Basic plotting
- Seaborn - Statistical visualizations
- Plotly - Interactive visualizations

#### Development and Testing
- Jupyter/IPykernel - Development environment
- Pylint - Code linting
- Pytest - Testing framework
- TQDM - Progress bars

#### Data Processing and Utilities
- Datasets - Data loading and management
- XLRD - Legacy Excel support
- Python-dotenv - Environment variables management

#### Web Development
- Reflex - Web framework

### Development Environment
- Visual Studio Code
- Jupyter Notebooks
- Git/GitHub
- Google Colab (Model training)
- Local development servers
- Docker containers

### Model Results
- Model Type: Neural Network 
- Training Accuracy: 75%
- Validation Accuracy: 50%
- Test Accuracy: 65%

## UI Screen shots.

Init Page

image.png

App Page


## Model Decision Reasoning
#### Why GELU over ReLU
- GELU has a smoother trasition gradiant
- GELU is better able to capture more complex data patterns which was needed for trying to understand language
- does not suffer from having neurons become inactive due to negative inputs. 

#### Why we chose softmax as out output activation.
- It is a standard approach to multi output/class classification.
- This also gave estimates for each output showing how strongly the text matched each category. This could be used to show how strongly the input would present in each classification.

#### Why we added an Attention Mechanism
- It added weighting to the model
- We used it before the dimesion reduction so that the effects would propigate through the rest of the layers.
- The attention model helps the sentence transformer (MiniLM) focus better on the relevant parts of the text

#### Why we decided to use MiniLM
- It simplified the embedding process. 
- Sped up training times
- Had a standardized system and account for more words than we had in our dataset.
- Had a standaried output of 384