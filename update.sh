#!/bin/bash

# Add all the log files
git add .

# Commit the changes with a timestamp
git commit -m "Log files for training run at $(date)"

# Push the changes to GitHub
git push origin master
