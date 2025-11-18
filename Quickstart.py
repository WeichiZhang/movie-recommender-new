from src.feature_extractor import MovieFeatureExtractor

# Initialize and process
extractor = MovieFeatureExtractor("data/ml-100k")
extractor.load_data()
extractor.extract_features()
extractor.consolidate_keywords()
encoded_features = extractor.encode_features()

# Use in your recommender system
feature_vector = extractor.get_feature_vector(1)  # Get features for movie ID 1
