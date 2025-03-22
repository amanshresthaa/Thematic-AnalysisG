# Complete Installation Guide for Thematic Analysis Program

This comprehensive guide will walk you through installing and setting up the Thematic Analysis program from scratch, including all prerequisites and dependencies.

## Prerequisites Installation

### 1. Install Git

#### For Windows:
1. Download Git from [Git for Windows](https://gitforwindows.org/)
2. Run the installer and use default settings (or customize if needed)
3. Verify installation:
```bash
git --version
```

#### For macOS:
1. Install Homebrew if not installed:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
2. Install Git:
```bash
brew install git
```

#### For Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install git
```

### 2. Install Python

#### For Windows:
1. Download Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
2. Run installer, CHECK "Add Python to PATH"
3. Verify installation:
```bash
python --version
pip --version
```

#### For macOS:
```bash
brew install python@3.9
```

#### For Linux:
```bash
sudo apt update
sudo apt install python3 python3-pip
```

### 3. Install Docker

#### For Windows:
1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Follow installation wizard
3. Start Docker Desktop

#### For macOS:
1. Install Docker Desktop:
```bash
brew install --cask docker
```
2. Start Docker Desktop from Applications

#### For Linux:
```bash
sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
```

## Project Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Thematic-AnalysisG.git
cd Thematic-AnalysisG
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Environment Setup

Create a `.env` file in the project root with:

```
VOYAGE_API_KEY=your_voyage_api_key
COHERE_API_KEY=your_cohere_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
```

### 5. Start Elasticsearch

```bash
# Pull Elasticsearch image
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.6.0

# Run Elasticsearch container
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.6.0
```

### 6. Create Required Directories

```bash
mkdir -p data/codebase_chunks
mkdir -p data/input
mkdir -p data/output
mkdir -p data/evaluation
mkdir -p data/training
mkdir -p data/optimized
mkdir -p documents
mkdir -p logs
```

## Running the Program

### 1. Verify Elasticsearch is Running

```bash
curl http://localhost:9200
```
You should see a JSON response with Elasticsearch information.

### 2. Start the Program

```bash
python src/main.py
```

## Usage Guide

### Input Data Preparation

1. Place your document files in the `documents/` directory
2. Supported formats:
   - Text files (.txt)
   - JSON files (.json)

### Program Components

The program runs through several analysis phases:
1. Quotation Analysis
2. Keyword Extraction
3. Coding Analysis
4. Grouping Analysis
5. Theme Development

Results are stored in:
- `data/output/` - Final analysis results
- `logs/` - Program execution logs

## Monitoring and Maintenance

### Check Logs
- `logs/info.log` - General information
- `logs/error.log` - Error messages
- `logs/debug.log` - Detailed debugging information

### Docker Container Management

```bash
# Check Elasticsearch status
docker ps

# View Elasticsearch logs
docker logs elasticsearch

# Stop Elasticsearch
docker stop elasticsearch

# Start Elasticsearch
docker start elasticsearch

# Remove container (if needed)
docker rm elasticsearch
```

## Troubleshooting

### Common Issues and Solutions

1. **Elasticsearch Connection Error**
   ```bash
   # Check if Elasticsearch is running
   docker ps | grep elasticsearch
   
   # Restart if needed
   docker restart elasticsearch
   ```

2. **Python Package Issues**
   ```bash
   # Reinstall requirements
   pip uninstall -r requirements.txt
   pip install -r requirements.txt
   ```

3. **Memory Issues**
   - Increase Docker memory allocation in Docker Desktop settings
   - For large documents, adjust Python memory:
   ```bash
   export PYTHONMEMOPTION="-Xmx8g"
   ```

4. **Port Conflicts**
   - Check if ports 9200/9300 are in use:
   ```bash
   # For Windows:
   netstat -ano | findstr 9200
   # For macOS/Linux:
   lsof -i :9200
   ```

### System Requirements

- CPU: 2+ cores recommended
- RAM: 8GB minimum, 16GB recommended
- Storage: 10GB free space
- OS: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)

## Updating the Program

```bash
# Get latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Restart Elasticsearch
docker restart elasticsearch
```

## Support

If you encounter issues:
1. Check the logs in `logs/` directory
2. Review the troubleshooting section above
3. Submit an issue on the GitHub repository
4. Contact the maintainer with logs and error details

---

For more detailed information about specific features, please refer to the documentation in the `docs/` directory.