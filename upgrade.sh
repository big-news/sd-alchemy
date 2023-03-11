#!/bin/bash

source venv/bin/activate

cd sd-scripts
git pull
pip install --use-pep517 --upgrade -r requirements.txt
