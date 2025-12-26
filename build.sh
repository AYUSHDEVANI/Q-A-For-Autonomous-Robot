#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Building..."

# 1. Install Python Dependencies
echo "Installing Python dependencies..."
# Install uv for faster dependency resolution
pip install uv

# Install dependencies using uv
uv pip install --system -r requirements.txt

# 2. Build Frontend
echo "Building Frontend..."
cd frontend
npm install
npm run build
cd ..

# 3. Move Build Artifacts to Backend Static/Templates
echo "Moving Frontend Artifacts..."

# Create directories if not exist
mkdir -p app/static/assets
mkdir -p app/templates

# Copy index.html to templates
cp frontend/dist/index.html app/templates/index.html

# Copy assets to static/assets
# Note: Vite builds to dist/assets. We need to copy contents of dist/assets to app/static/assets
# Or properly, copy dist/assets/* to app/static/assets/
cp -r frontend/dist/assets/* app/static/assets/

echo "Build Complete!"
