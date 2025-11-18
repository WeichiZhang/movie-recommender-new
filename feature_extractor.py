"""
MovieLens 100K Feature Extraction System
Extracts and encodes features for movie recommendation systems
"""

import pandas as pd
import numpy as np
import json
import re
from collections import defaultdict, Counter
import os
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieFeatureExtractor:
    """
    A comprehensive feature extraction system for MovieLens 100K dataset.
    
    This class handles:
    - Loading MovieLens 100K data
    - Extracting rich features (sub-genres, themes, moods, etc.)
    - Consolidating keywords into master lists
    - One-hot encoding for machine learning
    """
    
    def __init__(self, data_path: str = "data/ml-100k"):
        """
        Initialize the feature extractor.
        
        Args:
            data_path (str): Path to MovieLens 100K dataset
        """
        self.data_path = data_path
        self.movies_df = None
        self.movie_features = {}
        self.master_keywords = defaultdict(list)
        self.encoders = {}
        self.encoded_features = {}
        
    def load_data(self) -> bool:
        """
        Load MovieLens 100K dataset files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load movies data
            movies_file = os.path.join(self.data_path, 'u.item')
            self.movies_df = pd.read_csv(
                movies_file, 
                sep='|', 
                encoding='latin-1',
                header=None,
                names=[
                    'movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
                    'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
                ]
            )
            
            # Load ratings data if available
            ratings_file = os.path.join(self.data_path, 'u.data')
            if os.path.exists(ratings_file):
                self.ratings_df = pd.read_csv(
                    ratings_file,
                    sep='\t',
                    header=None,
                    names=['user_id', 'movie_id', 'rating', 'timestamp']
                )
                self.movie_ratings = self.ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
                self.movie_ratings.columns = ['movie_id', 'avg_rating', 'rating_count']
            else:
                logger.warning("Ratings data not found, proceeding without rating-based features")
                self.movie_ratings = None
                
            logger.info(f"Successfully loaded {len(self.movies_df)} movies")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def extract_features(self) -> Dict[int, Dict]:
        """
        Extract comprehensive features from all movies.
        
        Returns:
            Dict containing features for each movie
        """
        if self.movies_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        genre_columns = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        for _, movie in self.movies_df.iterrows():
            movie_id = movie['movie_id']
            features = self._extract_single_movie_features(movie, genre_columns)
            self.movie_features[movie_id] = features
            
        logger.info(f"Extracted features for {len(self.movie_features)} movies")
        return self.movie_features

    def _extract_single_movie_features(self, movie: pd.Series, genre_columns: List[str]) -> Dict[str, Any]:
        """Extract features for a single movie."""
        title = movie['title']
        
        # Extract basic features
        genres = self._extract_genres(movie, genre_columns)
        year = self._extract_year(title)
        
        return {
            'title': title,
            'year': year,
            'era': self._categorize_era(year),
            'genres': genres,
            'sub_genre': self._derive_sub_genres(genres, title),
            'themes': self._derive_themes(genres, title, year),
            'mood': self._derive_mood(genres),
            'target_audience': self._derive_target_audience(genres, title),
            'popularity': self._get_popularity(movie['movie_id'])
        }

    def _extract_genres(self, movie: pd.Series, genre_columns: List[str]) -> List[str]:
        return [genre for genre in genre_columns if movie[genre] == 1]

    def _extract_year(self, title: str) -> int:
        year_match = re.search(r'\((\d{4})\)', title)
        return int(year_match.group(1)) if year_match else 1900

    def _categorize_era(self, year: int) -> str:
        if year < 1960: return "Classic"
        elif year < 1970: return "60s"
        elif year < 1980: return "70s"
        elif year < 1990: return "80s"
        elif year < 2000: return "90s"
        else: return "Modern"

    def _get_popularity(self, movie_id: int) -> str:
        if self.movie_ratings is None:
            return "Unknown"
        rating_info = self.movie_ratings[self.movie_ratings['movie_id'] == movie_id]
        if len(rating_info) == 0:
            return "Unknown"
        avg_rating = rating_info['avg_rating'].iloc[0]
        if avg_rating >= 4.0: return "Very Popular"
        elif avg_rating >= 3.5: return "Popular"
        elif avg_rating >= 3.0: return "Average"
        else: return "Niche"

    def _derive_sub_genres(self, genres: List[str], title: str) -> List[str]:
        sub_genres = []
        title_lower = title.lower()
        
        # Create genre combinations
        if len(genres) >= 2:
            for i in range(len(genres)):
                for j in range(i + 1, len(genres)):
                    sub_genres.append(f"{genres[i]}-{genres[j]}")
        
        # Enhanced sub-genre detection
        enhanced_subgenres = {
            'Sci-Fi': [
                ('space', 'Space Opera'),
                ('alien', 'Alien Invasion'),
                ('cyber', 'Cyberpunk'),
                ('future', 'Future Dystopia')
            ],
            'Fantasy': [
                ('dragon', 'High Fantasy'),
                ('magic', 'Magical Fantasy'),
                ('fairy', 'Fairy Tale')
            ],
            'Horror': [
                ('slasher', 'Slasher Horror'),
                ('ghost', 'Supernatural Horror'),
                ('zombie', 'Zombie Horror')
            ]
        }
        
        for genre, patterns in enhanced_subgenres.items():
            if genre in genres:
                for keyword, subgenre in patterns:
                    if keyword in title_lower:
                        sub_genres.append(subgenre)
        
        return sub_genres if sub_genres else ['General']

    def _derive_themes(self, genres: List[str], title: str, year: int) -> List[str]:
        themes = set()
        
        # Genre-based themes
        theme_mapping = {
            'Action': ['Adventure', 'Heroism', 'Conflict'],
            'Romance': ['Love', 'Relationships', 'Passion'],
            'Drama': ['Emotional', 'Life Stories', 'Conflict'],
            'Comedy': ['Humor', 'Funny', 'Light-hearted'],
            'Horror': ['Fear', 'Suspense', 'Supernatural'],
            'Sci-Fi': ['Future', 'Technology', 'Space'],
            'Fantasy': ['Magic', 'Imagination', 'Mythology'],
            'Thriller': ['Suspense', 'Mystery', 'Tension'],
            'Crime': ['Justice', 'Criminal', 'Investigation'],
            'War': ['Conflict', 'Patriotism', 'History'],
            'Western': ['Frontier', 'Cowboys', 'Adventure']
        }
        
        for genre in genres:
            if genre in theme_mapping:
                themes.update(theme_mapping[genre])
        
        # Title-based themes
        title_themes = {
            'love': 'Love', 'heart': 'Love', 'romance': 'Love',
            'war': 'War', 'battle': 'War', 'soldier': 'War',
            'family': 'Family', 'children': 'Family',
            'school': 'Coming of Age', 'teen': 'Coming of Age',
            'friend': 'Friendship', 'buddy': 'Friendship'
        }
        
        title_lower = title.lower()
        for keyword, theme in title_themes.items():
            if keyword in title_lower:
                themes.add(theme)
        
        return sorted(list(themes))

    def _derive_mood(self, genres: List[str]) -> List[str]:
        mood_scores = {'Positive': 0, 'Intense': 0, 'Dark': 0, 'Thoughtful': 0}
        
        mood_mapping = {
            'Positive': ['Comedy', 'Musical', 'Animation', 'Children'],
            'Intense': ['Action', 'Adventure', 'Thriller'],
            'Dark': ['Horror', 'Film-Noir', 'Crime'],
            'Thoughtful': ['Drama', 'Documentary', 'War']
        }
        
        for mood, mood_genres in mood_mapping.items():
            for genre in genres:
                if genre in mood_genres:
                    mood_scores[mood] += 1
        
        # Return top 2 moods
        return [mood for mood, _ in sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)[:2]]

    def _derive_target_audience(self, genres: List[str], title: str) -> str:
        if 'Children' in genres or 'Animation' in genres:
            return "Family"
        elif 'Horror' in genres:
            return "Adult"
        elif 'Documentary' in genres:
            return "Educational"
        else:
            return "General"

    def consolidate_keywords(self) -> Dict[str, List[str]]:
        """Create master keyword lists for encoding."""
        all_sub_genres = set()
        all_themes = set()
        all_moods = set()
        all_audiences = set()
        all_eras = set()
        all_popularity = set()
        
        for features in self.movie_features.values():
            all_sub_genres.update(features['sub_genre'])
            all_themes.update(features['themes'])
            all_moods.update(features['mood'])
            all_audiences.add(features['target_audience'])
            all_eras.add(features['era'])
            all_popularity.add(features['popularity'])
        
        self.master_keywords = {
            'sub_genre': sorted(list(all_sub_genres)),
            'themes': sorted(list(all_themes)),
            'mood': sorted(list(all_moods)),
            'target_audience': sorted(list(all_audiences)),
            'era': sorted(list(all_eras)),
            'popularity': sorted(list(all_popularity))
        }
        
        logger.info("Master keywords consolidated")
        return self.master_keywords

    def encode_features(self) -> Dict[int, Dict[str, List[int]]]:
        """Convert features to one-hot encoded vectors."""
        if not self.master_keywords:
            raise ValueError("Call consolidate_keywords() first")
            
        # Create encoders
        self.encoders = {}
        for feature_type, keywords in self.master_keywords.items():
            self.encoders[feature_type] = {kw: i for i, kw in enumerate(keywords)}
        
        # Encode features
        for movie_id, features in self.movie_features.items():
            encoded = {}
            for feature_type in self.master_keywords.keys():
                vector = [0] * len(self.master_keywords[feature_type])
                feature_values = features[feature_type]
                if not isinstance(feature_values, list):
                    feature_values = [feature_values]
                
                for value in feature_values:
                    if value in self.encoders[feature_type]:
                        vector[self.encoders[feature_type][value]] = 1
                
                encoded[feature_type] = vector
            
            self.encoded_features[movie_id] = encoded
        
        logger.info(f"Encoded features for {len(self.encoded_features)} movies")
        return self.encoded_features

    def save_output(self, output_dir: str = "output") -> None:
        """Save all extracted features and encodings."""
        os.makedirs(output_dir, exist_ok=True)
        
        outputs = {
            'movie_features.json': self.movie_features,
            'encoded_features.json': self.encoded_features,
            'master_keywords.json': self.master_keywords,
            'encoders.json': self.encoders
        }
        
        for filename, data in outputs.items():
            with open(os.path.join(output_dir, filename), 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Output saved to {output_dir}/")

    def get_feature_vector(self, movie_id: int) -> Optional[Dict]:
        """Get encoded feature vector for a specific movie."""
        return self.encoded_features.get(movie_id)

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary statistics of extracted features."""
        return {
            'total_movies': len(self.movie_features),
            'feature_categories': len(self.master_keywords),
            'total_dimensions': sum(len(v) for v in self.master_keywords.values()),
            'category_sizes': {k: len(v) for k, v in self.master_keywords.items()}
        }
