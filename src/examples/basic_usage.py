"""
Basic usage example for MovieFeatureExtractor
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.feature_extractor import MovieFeatureExtractor

def main():
    # Initialize extractor
    extractor = MovieFeatureExtractor("data/ml-100k")
    
    # Load and process data
    extractor.load_data()
    extractor.extract_features()
    extractor.consolidate_keywords()
    extractor.encode_features()
    
    # Save results
    extractor.save_output("output")
    
    # Display summary
    summary = extractor.get_feature_summary()
    print("Feature Extraction Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
