#!/bin/bash

logdir="logs"

source venv/bin/activate
tensorboard --logdir=${logdir} --port=6006