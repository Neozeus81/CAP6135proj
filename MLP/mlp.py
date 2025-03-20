import sys
import os

# Get the absolute path to the directory containing the file to import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
file_dir = os.path.join(parent_dir, 'data')

sys.path.append(file_dir)

import data
