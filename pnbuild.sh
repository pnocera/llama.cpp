#!/bin/bash

# Version file
VERSION_FILE=".version"

# Image name
IMAGE_NAME="quay.io/pnocera/llama-vulkan-radv"

# Read current version or initialize
if [ -f "$VERSION_FILE" ]; then
    VERSION=$(cat "$VERSION_FILE")
else
    VERSION="1.0.0"
fi

# Parse version components
IFS='.' read -r major minor patch <<< "$VERSION"

# Increment patch version
patch=$((patch + 1))
NEW_VERSION="${major}.${minor}.${patch}"

echo "Building image with version ${NEW_VERSION}..."

# Build the image with version tag
if podman build --no-cache -t ${IMAGE_NAME}:${NEW_VERSION} -f Dockerfile.vulkan-radv .; then
    echo "Build successful!"
    
    # Tag as latest
    echo "Tagging as latest..."
    podman tag ${IMAGE_NAME}:${NEW_VERSION} ${IMAGE_NAME}:latest
    
    # Push both tags
    echo "Pushing version ${NEW_VERSION}..."
    if podman push ${IMAGE_NAME}:${NEW_VERSION}; then
        echo "Pushing latest..."
        if podman push ${IMAGE_NAME}:latest; then
            # Save new version only if everything succeeded
            echo "$NEW_VERSION" > "$VERSION_FILE"
            echo "Successfully pushed ${IMAGE_NAME}:${NEW_VERSION} and ${IMAGE_NAME}:latest"
            echo "Version updated to ${NEW_VERSION}"
        else
            echo "Failed to push latest tag"
            exit 1
        fi
    else
        echo "Failed to push version tag"
        exit 1
    fi
else
    echo "Build failed!"
    exit 1
fi