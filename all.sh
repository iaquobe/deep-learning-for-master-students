#!/bin/bash

python -m eurosat.rgb.train
python -m eurosat.rgb.test

python -m eurosat.ms.train
python -m eurosat.ms.test
