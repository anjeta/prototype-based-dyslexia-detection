"""
Created on Wed Jan 27 23:34:19 2021

@author: Aneta

Script for visualizating training logs.

"""

# Import the JSON module to work with json files
import json
# Import other libraries
from visualizations import visualize_logs

def main():
    results_dir = "../checkpoint/"
    with open(results_dir + 'checkpoint_logs.json') as f:
        logs = json.load(f)
    visualize_logs(logs)

if __name__ == "__main__":
    main()