# part1_core.py
import random
import shutil
from collections import defaultdict

import torch.nn as nn
import torchvision.transforms as transforms
import timm
import json

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List, Tuple
import cv2
import tempfile
from datetime import datetime
import opennsfw2  as n2

import logging
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import torch
from typing import Dict, Tuple, List
from PIL import Image
import torch.nn.functional as F
from typing import List, Dict, Set, Optional
import json
from pathlib import Path
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    # Model Configuration
    MODEL_NAME = "efficientnet_b0"
    MODEL_ARCHITECTURE = "efficientnet_b0"
    PRETRAINED = True
    FREEZE_BACKBONE = False

    # Training Parameters
    YOLO_WEIGHTS = "yolov5s.pt"
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    IMG_SIZE = 640
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Content Categories
    SEVERITY_LEVELS = {
        'extreme': ['explicit_nudity', 'graphic_violence', 'gore'],
        'moderate': ['partial_nudity', 'suggestive', 'mild_violence'],
        'mild': ['revealing_clothes', 'implied_suggestive'],
        'safe': ['fully_clothed', 'non_suggestive']
    }

    # Original body attributes and categories
    BODY_ATTRIBUTES = {
        'proportions': {
            'chest': ['small', 'medium', 'large', 'xlarge'],
            'hips': ['small', 'medium', 'large', 'xlarge'],
            'buttocks': ['small', 'medium', 'large', 'xlarge'],
            'waist': ['small', 'medium', 'large'],
            'thighs': ['small', 'medium', 'large'],
            'arms': ['small', 'medium', 'large'],
            'legs': ['small', 'medium', 'large'],
            'breast_size': ['small', 'medium', 'large', 'xlarge'],
            'ass_size': ['small', 'medium', 'large', 'xlarge'],
        },
        'exposure_level': {
            'chest': ['covered', 'partial', 'exposed', 'full'],
            'back': ['covered', 'partial', 'exposed', 'full'],
            'pelvic': ['covered', 'partial', 'exposed', 'full'],
            'buttocks': ['covered', 'partial', 'exposed', 'full'],
            'legs': ['covered', 'partial', 'exposed', 'full'],
            'midriff': ['covered', 'partial', 'exposed', 'full'],
            'shoulders': ['covered', 'partial', 'exposed'],
            'exposed_breasts': ['covered', 'partial_exposure', 'full_exposure'],
            'exposed_ass': ['covered', 'partial_exposure', 'full_exposure'],
            'exposed_genitalia': ['covered', 'partial_exposure', 'full_exposure'],
        },
        'coverage_type': {
            'upper_body': ['full', 'partial', 'minimal', 'none'],
            'lower_body': ['full', 'partial', 'minimal', 'none'],
            'rear': ['full', 'partial', 'minimal', 'none'],
            'midsection': ['full', 'partial', 'minimal', 'none']
        }
    }

    CATEGORIES = {
        'art_style': [
            'cel_shaded', 'digital', 'watercolor', 'sketch',
            'suggestive_style', 'erotic_art', 'pinup_style'
        ],
        'character_pose': [
            'standing', 'sitting', 'running', 'profile',
            'suggestive_pose', 'intimate_pose', 'explicit_pose'
        ],
        'scene_type': [
            'indoor', 'outdoor', 'bedroom', 'bathroom', 'dungeon',
            'school', 'office', 'fantasy_setting'
        ],
        'emotion': ['happy', 'serious', 'surprised', 'neutral', 'seductive'],
        'clothing': [
            'casual', 'formal', 'sportswear', 'swimwear',
            'lingerie', 'partial', 'minimal', 'none'
        ],
        'clothing_coverage': [
            'full', 'moderate', 'revealing', 'minimal', 'none'
        ],
        'content_rating': ['safe', 'suggestive', 'questionable', 'explicit'],
        'age_rating': ['all_ages', 'teen', 'mature', 'adult'],
        'safety_tags': [
            'safe', 'suggestive', 'questionable',
            'nsfw', 'explicit', 'blocked'
        ]
    }

    # Add orientation detection settings
    ORIENTATION_DETECTION = {
        'confidence_threshold': 0.7,
        'min_visibility_score': 0.5,
        'orientation_categories': ['front', 'back', 'side', 'three_quarter'],
        'pose_attributes': ['facing_direction', 'head_position', 'body_angle', 'visibility']
    }

    # Thresholds and Validation
    CONFIDENCE_THRESHOLDS = {
        'content_rating': 0.85,
        'age_rating': 0.85,
        'clothing': 0.8,
        'clothing_coverage': 0.8,
        'clothing_style': 0.8,
        'character_pose': 0.8,
        'art_style': 0.75,
        'background': 0.75,
        'background_context': 0.8,
        'safety_tags': 0.9,
        'composition': 0.75,
        'default': 0.5
    }

    DETECTION_THRESHOLDS = {
        'proportions': 0.85,
        'exposure_level': 0.90,
        'coverage_type': 0.85,
        'exposure_edge_cases': 0.92,
        'body_orientations': 0.88,
        'motion_indicators': 0.95,
        'implied_content': 0.93
    }

    # Model Configuration
    MODEL_CONFIG = {
        'input_size': (224, 224),
        'num_channels': 3,
        'dropout_rate': 0.2,
        'activation': 'relu',
        'pooling': 'avg',
        'use_batch_norm': True
    }

    CHARACTER_DETECTION = {
        'confidence_threshold': 0.7,
        'max_characters': 10,
        'min_size': 64,
    }

    # Add character attributes to CATEGORIES
    CATEGORIES.update({
        'character_gender': ['female', 'male', 'ambiguous'],
        'character_hair_color': ['blonde', 'brown', 'black', 'blue', 'pink', 'red', 'white', 'other'],
        'character_eye_color': ['blue', 'brown', 'green', 'red', 'purple', 'other'],
        'character_hair_style': ['long', 'short', 'twin_tails', 'ponytail', 'other'],
        'character_age': ['young', 'teen', 'adult', 'elderly'],
        'character_body_type': ['slim', 'average', 'muscular', 'plus_size'],
        'character_species': ['human', 'animal_ears', 'demon', 'angel', 'monster', 'other']
    })


    # Checkpoint Configuration
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = "best_model.pth"
    RESUME_TRAINING = True


class CharacterDatabase:
    """Database for storing and matching anime characters"""

    def __init__(self, database_path: Optional[Path] = None):
        self.characters = {}
        self.feature_dim = 512  # Dimension of feature vectors
        self.index = faiss.IndexFlatIP(self.feature_dim)  # Inner product index
        self.character_ids = []
        self.database_path = database_path or Path("character_database.pkl")

        # Load existing database if available
        self.load_database()

        # Character metadata structure
        self.character_metadata = {
            'name': str,
            'series': str,
            'hair_color': str,
            'eye_color': str,
            'gender': str,
            'age': str,
            'features': List[str],
            'relationships': Dict[str, str],
            'variants': List[str]
        }

    def load_database(self):
        """Load character database from file"""
        try:
            if self.database_path.exists():
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.characters = data['characters']
                    self.character_ids = data['character_ids']
                    features = data['features']
                    self.index = faiss.IndexFlatIP(self.feature_dim)
                    if len(features) > 0:
                        self.index.add(np.array(features))
        except Exception as e:
            logger.error(f"Error loading character database: {str(e)}")
            self.characters = {}
            self.character_ids = []
            self.index = faiss.IndexFlatIP(self.feature_dim)

    def save_database(self):
        """Save character database to file"""
        try:
            features = []
            if self.index.ntotal > 0:
                features = self.index.reconstruct_n(0, self.index.ntotal)

            data = {
                'characters': self.characters,
                'character_ids': self.character_ids,
                'features': features
            }
            with open(self.database_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving character database: {str(e)}")

    def add_character(self, character_id: str, metadata: Dict, feature_vector: np.ndarray):
        """Add a new character to the database"""
        if character_id not in self.characters:
            self.characters[character_id] = metadata
            self.character_ids.append(character_id)
            self.index.add(feature_vector.reshape(1, -1))
            self.save_database()

    def search_characters(self, feature_vector: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar characters using feature vector"""
        feature_vector = feature_vector.reshape(1, -1)
        D, I = self.index.search(feature_vector, k)

        results = []
        for idx, sim_score in zip(I[0], D[0]):
            if idx < len(self.character_ids):
                char_id = self.character_ids[idx]
                char_data = self.characters[char_id].copy()
                char_data['similarity_score'] = float(sim_score)
                results.append(char_data)

        return results


class CharacterMatcher:
    """Enhanced character detection and matching system"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.feature_extractor = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

        self.detector.to(self.device)
        self.feature_extractor.to(self.device)

        self.detector.eval()
        self.feature_extractor.eval()

        self.character_db = CharacterDatabase()

        # Load character embeddings and features
        self.load_character_data()

        self.logger = logging.getLogger(__name__)

    def load_character_data(self):
        """Load character data and embeddings"""
        try:
            character_data_path = Path("character_data.json")
            if character_data_path.exists():
                with open(character_data_path, 'r') as f:
                    self.character_data = json.load(f)
            else:
                self.character_data = self._initialize_character_data()

        except Exception as e:
            self.logger.error(f"Error loading character data: {str(e)}")
            self.character_data = self._initialize_character_data()

    def _initialize_character_data(self) -> Dict:
        """Initialize basic character data structure"""
        return {
            'characters': {
                # Example character entry
                'character_1': {
                    'name': 'Character Name',
                    'series': 'Series Name',
                    'features': {
                        'hair_color': ['blonde'],
                        'eye_color': ['blue'],
                        'distinctive_features': ['twin_tails', 'hair_ribbon'],
                    },
                    'relationships': {
                        'character_2': 'friend',
                        'character_3': 'rival'
                    },
                    'variants': ['school_uniform', 'casual', 'battle']
                }
            },
            'series': {
                'Series Name': {
                    'characters': ['character_1'],
                    'genre': ['action', 'romance'],
                    'setting': 'modern'
                }
            }
        }

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract feature vector from character image"""
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            image_tensor = transform(image).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)

            return features.cpu().numpy()

        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.zeros((1, 512))  # Return zero vector on error

    def detect_and_match_characters(self, image_path: Path) -> List[Dict]:
        """Detect and match characters in an image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess_image(image)

            # Detect characters
            with torch.no_grad():
                detections = self.detector(image_tensor.to(self.device))[0]

            matched_characters = []

            # Process each detection
            for box, score in zip(detections['boxes'], detections['scores']):
                if score > Config.CHARACTER_DETECTION['confidence_threshold']:
                    # Extract character region
                    x1, y1, x2, y2 = map(int, box.tolist())
                    char_image = image.crop((x1, y1, x2, y2))

                    # Extract features
                    features = self.extract_features(char_image)

                    # Match character
                    matches = self.character_db.search_characters(features)

                    # Add detection info
                    char_info = {
                        'box': box.tolist(),
                        'confidence': float(score),
                        'matches': matches
                    }

                    matched_characters.append(char_info)

            return matched_characters

        except Exception as e:
            self.logger.error(f"Error in character detection and matching: {str(e)}")
            return []

    def add_character_to_database(self,
                                  character_id: str,
                                  metadata: Dict,
                                  reference_images: List[Path]):
        """Add a new character to the database with reference images"""
        try:
            # Extract and average features from reference images
            features = []
            for img_path in reference_images:
                image = Image.open(img_path).convert('RGB')
                feat = self.extract_features(image)
                features.append(feat)

            if features:
                avg_feature = np.mean(features, axis=0)
                self.character_db.add_character(character_id, metadata, avg_feature)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error adding character to database: {str(e)}")
            return False


# Update ContentAnalyzer to include character matching
class ContentAnalyzer:
    def __init__(self):
        self.nsfw_detector = NSFWDetector()
        self.character_matcher = CharacterMatcher()
        self.logger = logging.getLogger(__name__)

    def analyze_content(self, image_path: Path) -> Dict[str, Dict[str, float]]:
        """Comprehensive content analysis including character matching"""
        scores = {
            # ... (existing scores remain the same)
            'characters': [],
            'character_matches': []
        }

        try:
            # Existing content analysis
            is_nsfw, nsfw_score = self.nsfw_detector.check_content(image_path)
            self._update_scores(scores, nsfw_score)

            # Add character detection and matching
            matched_characters = self.character_matcher.detect_and_match_characters(image_path)
            scores['character_matches'] = matched_characters

            # Update scores based on character information
            self._update_scores_with_characters(scores, matched_characters)

            self._normalize_scores(scores)

        except Exception as e:
            self.logger.error(f"Error analyzing content for {image_path}: {str(e)}")

        return scores



class CharacterDetector:
    """Handles anime character detection in images"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Character types/attributes we want to detect
        self.character_attributes = {
            'gender': ['female', 'male', 'ambiguous'],
            'hair_color': ['blonde', 'brown', 'black', 'blue', 'pink', 'red', 'white', 'other'],
            'eye_color': ['blue', 'brown', 'green', 'red', 'purple', 'other'],
            'hair_style': ['long', 'short', 'twin_tails', 'ponytail', 'other'],
            'age_appearance': ['young', 'teen', 'adult', 'elderly'],
            'body_type': ['slim', 'average', 'muscular', 'plus_size'],
            'species': ['human', 'animal_ears', 'demon', 'angel', 'monster', 'other']
        }

        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for character detection"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to tensor and normalize
        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        return image_tensor.unsqueeze(0)

    def detect_characters(self, image_path: Path) -> List[Dict[str, Dict[str, float]]]:
        """Detect and analyze characters in an image"""
        try:
            # Load and preprocess image
            with Image.open(image_path) as img:
                image_tensor = self.preprocess_image(img)
                image_tensor = image_tensor.to(self.device)

            # Get character detections
            with torch.no_grad():
                predictions = self.model(image_tensor)

            characters = []
            for box, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
                if score > 0.7:  # Confidence threshold
                    character_info = self._analyze_character_region(img, box)
                    characters.append(character_info)

            return characters

        except Exception as e:
            self.logger.error(f"Error detecting characters in {image_path}: {str(e)}")
            return []

    def _analyze_character_region(self, image: Image.Image, box: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """Analyze detected character region for attributes"""
        attributes = {}

        # Convert box coordinates to integers
        x1, y1, x2, y2 = map(int, box.tolist())

        # Crop character region
        character_region = image.crop((x1, y1, x2, y2))

        # Analyze each attribute category
        for category, possible_values in self.character_attributes.items():
            attributes[category] = self._predict_attribute(character_region, category, possible_values)

        return attributes

    def _predict_attribute(self, region: Image.Image, category: str, possible_values: List[str]) -> Dict[str, float]:
        """Predict probabilities for character attributes"""
        # This is a placeholder for actual attribute prediction
        # In a real implementation, you would use specific models for each attribute
        scores = torch.softmax(torch.randn(len(possible_values)), dim=0)
        return {value: float(score) for value, score in zip(possible_values, scores)}


class ImagePreprocessor:
    def __init__(self, target_size: int = Config.IMG_SIZE):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

class RobustImageDataset(Dataset):
    def __init__(self, image_dir: str, labels_file: Optional[str] = None, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.data = []
        self._content_analyzer = None

        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")

        if labels_file and Path(labels_file).exists():
            with open(labels_file, 'r') as f:
                self.data = json.load(f)['images']
        else:
            image_files = self._find_images_recursive(self.image_dir)
            if not image_files:
                raise ValueError(f"No valid images found in directory: {image_dir}")
            logger.info(f"Found {len(image_files)} images in {image_dir}")
            for img_path in image_files:
                self.data.append({
                    'file_name': str(img_path.relative_to(self.image_dir)),
                    'labels': None
                })

    def _get_content_analyzer(self):
        if self._content_analyzer is None:
            self._content_analyzer = ContentAnalyzer()
        return self._content_analyzer

    def _find_images_recursive(self, directory: Path) -> List[Path]:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
        image_files = []
        try:
            for item in directory.rglob('*'):
                if (item.is_file() and
                        item.suffix.lower() in valid_extensions and
                        not item.name.startswith('.')):
                    image_files.append(item)
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {str(e)}")
        return image_files

    def _generate_labels(self, content_scores: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        # Label generation logic remains the same as before...
        pass  # Include full implementation here

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self.image_dir / item['file_name']

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            image = torch.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE))

        if item['labels'] is None:
            try:
                analyzer = self._get_content_analyzer()
                content_scores = analyzer.analyze_content(image_path)
                item['labels'] = self._generate_labels(content_scores)
            except Exception as e:
                logger.error(f"Error analyzing content for {image_path}: {str(e)}")
                item['labels'] = self._get_default_labels()

        tensor_labels = {}
        for category, label in item['labels'].items():
            if category in Config.CATEGORIES:
                try:
                    if label not in Config.CATEGORIES[category]:
                        label = Config.CATEGORIES[category][0]
                    label_idx = Config.CATEGORIES[category].index(label)
                    tensor_labels[category] = torch.tensor(label_idx, dtype=torch.long)
                except Exception as e:
                    tensor_labels[category] = torch.tensor(0, dtype=torch.long)

        return {
            'image': image,
            'file_name': item['file_name'],
            'labels': tensor_labels
        }


class RobustImageClassifier(nn.Module):
    def __init__(self, num_classes_dict):
        super().__init__()
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        feat_dim = self.backbone.num_features
        self.classifiers = nn.ModuleDict()

        sensitive_categories = ['content_rating', 'nsfw_attributes', 'age_rating',
                                'clothing', 'character_pose', 'art_style']

        for category, num_classes in num_classes_dict.items():
            if category in sensitive_categories:
                self.classifiers[category] = nn.Sequential(
                    nn.BatchNorm1d(feat_dim),
                    nn.Linear(feat_dim, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.BatchNorm1d(1024),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.BatchNorm1d(512),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            else:
                self.classifiers[category] = nn.Sequential(
                    nn.BatchNorm1d(feat_dim),
                    nn.Linear(feat_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return {
            category: classifier(features)
            for category, classifier in self.classifiers.items()
        }

    def predict_proba(self, x):
        outputs = self.forward(x)
        return {
            category: torch.softmax(output, dim=1)
            for category, output in outputs.items()
        }


class NSFWDetector:
    """Handles NSFW content detection using OpenNSFW2"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._setup_logging()
        self._setup_model()

    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('nsfw_detector.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_model(self):
        """Initialize the NSFW detection model"""
        try:
            self.model = n2.make_open_nsfw_model()
            self.logger.info("NSFW model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize NSFW model: {str(e)}")
            raise

    def _process_image(self, image: Image.Image) -> float:
        """Process a single PIL Image and return NSFW score"""
        try:
            processed_image = n2.preprocess_image(image, n2.Preprocessing.YAHOO)
            input_data = np.expand_dims(processed_image, axis=0)
            predictions = self.model.predict(input_data, verbose=0)
            return float(predictions[0][1])
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return 1.0

    def _process_frames(self, frames: List[Image.Image]) -> List[float]:
        """Process multiple frames in batch"""
        try:
            processed_frames = [n2.preprocess_image(frame, n2.Preprocessing.YAHOO) for frame in frames]
            batch = np.stack(processed_frames)
            predictions = self.model.predict(batch, verbose=0)
            return [float(pred[1]) for pred in predictions]
        except Exception as e:
            self.logger.error(f"Error processing frames: {str(e)}")
            return [1.0] * len(frames)

    def check_gif(self, gif_path: Path) -> Tuple[bool, float]:
        """Check if a GIF contains NSFW content"""
        try:
            with Image.open(str(gif_path)) as gif:
                frames = []
                frame_count = 0

                while frame_count < 50:
                    try:
                        frame = gif.convert('RGB')
                        frames.append(frame.copy())
                        frame_count += 1
                        gif.seek(gif.tell() + 1)
                    except EOFError:
                        break

                if not frames:
                    return True, 1.0

                batch_size = 16
                all_scores = []

                for i in range(0, len(frames), batch_size):
                    batch_frames = frames[i:i + batch_size]
                    scores = self._process_frames(batch_frames)
                    all_scores.extend(scores)

                max_score = max(all_scores)
                avg_score = sum(all_scores) / len(all_scores)
                high_scores = sum(1 for score in all_scores if score > self.threshold)

                is_nsfw = (max_score > self.threshold or
                           avg_score > self.threshold * 0.8 or
                           high_scores >= 2)

                return is_nsfw, max_score

        except Exception as e:
            self.logger.error(f"Error analyzing GIF {gif_path}: {str(e)}")
            return True, 1.0

    def check_image(self, image_path: Path) -> Tuple[bool, float]:
        """Check if a static image contains NSFW content"""
        try:
            nsfw_score = n2.predict_image(str(image_path))
            return nsfw_score > self.threshold, nsfw_score
        except Exception as e:
            self.logger.error(f"Error checking image {image_path}: {str(e)}")
            return True, 1.0

    def check_content(self, file_path: Path) -> Tuple[bool, float]:
        """Universal checker that handles both static images and GIFs"""
        try:
            with Image.open(str(file_path)) as img:
                is_gif = getattr(img, "is_animated", False)

            if is_gif:
                return self.check_gif(file_path)
            else:
                return self.check_image(file_path)
        except Exception as e:
            self.logger.error(f"Error determining file type: {str(e)}")
            return True, 1.0


class CharacterOrientation:
    """Handles character pose and orientation detection"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load pose estimation model for orientation detection
        self.pose_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=4)  # 4 orientations
        self.pose_model.to(self.device)
        self.pose_model.eval()

        # Define orientation categories
        self.orientations = {
            'front': 0,
            'back': 1,
            'side': 2,
            'three_quarter': 3
        }

        # Detailed pose attributes
        self.pose_attributes = {
            'facing_direction': ['front', 'back', 'left_side', 'right_side', 'three_quarter_left',
                                 'three_quarter_right'],
            'head_position': ['straight', 'turned_left', 'turned_right', 'looking_up', 'looking_down'],
            'body_angle': ['straight', 'twisted', 'bent', 'leaning'],
            'visibility': {
                'face': ['full', 'partial', 'hidden'],
                'torso': ['front', 'back', 'side'],
                'limbs': ['all_visible', 'partially_hidden', 'mostly_hidden']
            }
        }

        self.logger = logging.getLogger(__name__)

    def detect_orientation(self, image: Image.Image) -> Dict[str, Dict[str, float]]:
        """Detect character orientation and pose details"""
        try:
            # Transform image for pose model
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            image_tensor = transform(image).unsqueeze(0).to(self.device)

            # Get orientation predictions
            with torch.no_grad():
                orientation_scores = torch.softmax(self.pose_model(image_tensor), dim=1)

            # Convert to dictionary of probabilities
            orientation_probs = {
                orient: float(orientation_scores[0][idx])
                for orient, idx in self.orientations.items()
            }

            # Analyze detailed pose attributes
            pose_details = self._analyze_pose_details(image)

            return {
                'orientation': orientation_probs,
                'pose_details': pose_details
            }

        except Exception as e:
            self.logger.error(f"Error detecting orientation: {str(e)}")
            return self._get_default_orientation()

    def _analyze_pose_details(self, image: Image.Image) -> Dict[str, Dict[str, float]]:
        """Analyze detailed pose attributes"""
        try:
            # Initialize pose analysis results
            pose_details = {}

            # Analyze facing direction
            pose_details['facing_direction'] = self._analyze_facing_direction(image)

            # Analyze head position
            pose_details['head_position'] = self._analyze_head_position(image)

            # Analyze body angle
            pose_details['body_angle'] = self._analyze_body_angle(image)

            # Analyze visibility
            pose_details['visibility'] = self._analyze_visibility(image)

            return pose_details

        except Exception as e:
            self.logger.error(f"Error analyzing pose details: {str(e)}")
            return self._get_default_pose_details()

    def _analyze_facing_direction(self, image: Image.Image) -> Dict[str, float]:
        """Analyze character's facing direction"""
        # Use facial landmarks and body keypoints to determine facing direction
        scores = {}
        for direction in self.pose_attributes['facing_direction']:
            # Implement actual detection logic here
            # For now, using placeholder probabilities
            scores[direction] = 0.0

        # Set highest probability for most likely direction
        scores[self._detect_primary_direction(image)] = 0.8
        return scores

    def _analyze_head_position(self, image: Image.Image) -> Dict[str, float]:
        """Analyze character's head position"""
        scores = {}
        for position in self.pose_attributes['head_position']:
            scores[position] = 0.0

        # Set highest probability for detected head position
        scores[self._detect_head_position(image)] = 0.8
        return scores

    def _analyze_body_angle(self, image: Image.Image) -> Dict[str, float]:
        """Analyze character's body angle"""
        scores = {}
        for angle in self.pose_attributes['body_angle']:
            scores[angle] = 0.0

        # Set highest probability for detected body angle
        scores[self._detect_body_angle(image)] = 0.8
        return scores

    def _analyze_visibility(self, image: Image.Image) -> Dict[str, Dict[str, float]]:
        """Analyze visibility of different body parts"""
        visibility = {}
        for part, states in self.pose_attributes['visibility'].items():
            visibility[part] = {state: 0.0 for state in states}
            # Set highest probability for detected visibility state
            visibility[part][self._detect_visibility(image, part)] = 0.8
        return visibility

    def _detect_primary_direction(self, image: Image.Image) -> str:
        """Detect primary facing direction of character"""
        # Implement actual detection logic here
        # For now, returning a default value
        return 'front'

    def _detect_head_position(self, image: Image.Image) -> str:
        """Detect head position of character"""
        return 'straight'

    def _detect_body_angle(self, image: Image.Image) -> str:
        """Detect body angle of character"""
        return 'straight'

    def _detect_visibility(self, image: Image.Image, part: str) -> str:
        """Detect visibility state of specific body part"""
        return self.pose_attributes['visibility'][part][0]

    def _get_default_orientation(self) -> Dict[str, Dict[str, float]]:
        """Return default orientation probabilities"""
        return {
            'orientation': {orient: 0.25 for orient in self.orientations},
            'pose_details': self._get_default_pose_details()
        }

    def _get_default_pose_details(self) -> Dict[str, Dict[str, float]]:
        """Return default pose details"""
        return {
            'facing_direction': {dir: 1 / len(self.pose_attributes['facing_direction'])
                                 for dir in self.pose_attributes['facing_direction']},
            'head_position': {pos: 1 / len(self.pose_attributes['head_position'])
                              for pos in self.pose_attributes['head_position']},
            'body_angle': {angle: 1 / len(self.pose_attributes['body_angle'])
                           for angle in self.pose_attributes['body_angle']},
            'visibility': {
                part: {state: 1 / len(states) for state in states}
                for part, states in self.pose_attributes['visibility'].items()
            }
        }


# Update CharacterMatcher to include orientation detection
class CharacterMatcher:
    def __init__(self):
        # ... (previous initialization code remains the same)
        self.orientation_detector = CharacterOrientation()

    def detect_and_match_characters(self, image_path: Path) -> List[Dict]:
        """Detect, match, and analyze orientation of characters in an image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess_image(image)

            # Detect characters
            with torch.no_grad():
                detections = self.detector(image_tensor.to(self.device))[0]

            matched_characters = []

            # Process each detection
            for box, score in zip(detections['boxes'], detections['scores']):
                if score > Config.CHARACTER_DETECTION['confidence_threshold']:
                    # Extract character region
                    x1, y1, x2, y2 = map(int, box.tolist())
                    char_image = image.crop((x1, y1, x2, y2))

                    # Extract features
                    features = self.extract_features(char_image)

                    # Match character
                    matches = self.character_db.search_characters(features)

                    # Detect orientation
                    orientation_data = self.orientation_detector.detect_orientation(char_image)

                    # Add detection info
                    char_info = {
                        'box': box.tolist(),
                        'confidence': float(score),
                        'matches': matches,
                        'orientation': orientation_data
                    }

                    matched_characters.append(char_info)

            return matched_characters

        except Exception as e:
            self.logger.error(f"Error in character detection and matching: {str(e)}")
            return []

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=0.01
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        self.criterion = {}
        sensitive_categories = ['content_rating', 'nsfw_attributes', 'age_rating',
                                'clothing', 'character_pose', 'art_style']

        for category in Config.CATEGORIES.keys():
            if category in sensitive_categories:
                self.criterion[category] = nn.CrossEntropyLoss(label_smoothing=0.1)
            else:
                self.criterion[category] = nn.CrossEntropyLoss()

    def train(self, num_epochs):
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(num_epochs):
            train_loss = self._train_epoch()
            val_loss, val_metrics = self._validate()
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                torch.save(best_model_state, 'best_model.pth')

            self._log_metrics(epoch, train_loss, val_loss, val_metrics)

        return best_model_state

    def _train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch['image'].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = sum(self.criterion[cat](outputs[cat], labels[cat])
                       for cat in outputs.keys())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        total_loss = 0
        metrics = {category: {'correct': 0, 'total': 0}
                   for category in Config.CATEGORIES}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}

                outputs = self.model(images)
                loss = sum(self.criterion[cat](outputs[cat], labels[cat])
                           for cat in outputs.keys())
                total_loss += loss.item()

                for category in outputs:
                    _, preds = torch.max(outputs[category], 1)
                    metrics[category]['correct'] += (preds == labels[category]).sum().item()
                    metrics[category]['total'] += labels[category].size(0)

        val_loss = total_loss / len(self.val_loader)
        accuracies = {
            category: metrics[category]['correct'] / metrics[category]['total']
            for category in metrics
        }

        return val_loss, accuracies


def custom_collate_fn(batch):
    if len(batch) == 0:
        return {}

    collated = {
        'image': [],
        'file_name': [],
        'labels': defaultdict(list)
    }

    for item in batch:
        if item['image'] is not None:
            collated['image'].append(item['image'])
            collated['file_name'].append(item['file_name'])
            for category, tensor in item['labels'].items():
                if tensor is not None:
                    collated['labels'][category].append(tensor)

    if collated['image']:
        collated['image'] = torch.stack(collated['image'])
        for category in collated['labels']:
            if collated['labels'][category]:
                collated['labels'][category] = torch.stack(collated['labels'][category])

    return collated


def setup_directories(input_dir: Path, train_dir: Path, val_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Set up training and validation directories with images from input directory"""
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing files
    for dir_path in [train_dir, val_dir]:
        for file in dir_path.glob("*.[jp][pn][g]"):
            file.unlink()

    input_images = list(input_dir.glob("*.[jp][pn][g]"))
    if not input_images:
        raise ValueError(f"No images found in input directory: {input_dir}")

    # Split images
    random.shuffle(input_images)
    split_idx = int(len(input_images) * 0.8)  # 80% training, 20% validation
    train_images = input_images[:split_idx]
    val_images = input_images[split_idx:]

    # Copy images to respective directories
    for img_path in tqdm(train_images, desc="Copying training images"):
        shutil.copy2(img_path, train_dir / img_path.name)

    for img_path in tqdm(val_images, desc="Copying validation images"):
        shutil.copy2(img_path, val_dir / img_path.name)

    return train_images, val_images


def main():
    logging.basicConfig(level=logging.INFO)
    results_dir = Path('results') / datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path("./hentai")
    train_dir = Path("./training-hentai")
    val_dir = Path("./validated-hentai")
    model_path = "model.pth"
    character_db_path = results_dir / "character_database.pkl"

    try:
        # Initialize components
        preprocessor = ImagePreprocessor()
        nsfw_detector = NSFWDetector()
        content_analyzer = ContentAnalyzer()
        character_matcher = CharacterMatcher()

        # Load or create character database
        if character_db_path.exists():
            character_matcher.character_db.load_database()
            logger.info("Loaded existing character database")
        else:
            logger.info("Creating new character database")

        # Process and sort images
        logger.info("Processing images from input directory...")
        input_images = list(input_dir.glob('*.[jp][pn][g]'))
        val_split = 0.2

        # Create directories if they don't exist
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Dictionary to store character statistics
        character_stats = {
            'total_characters_detected': 0,
            'characters_by_orientation': {
                'front': 0,
                'back': 0,
                'side': 0,
                'three_quarter': 0
            },
            'character_matches': {},
            'orientation_distribution': {}
        }

        for img_path in tqdm(input_images, desc="Processing images"):
            try:
                # Check NSFW content
                is_nsfw, nsfw_score = nsfw_detector.check_content(img_path)

                # Analyze content including character detection
                content_scores = content_analyzer.analyze_content(img_path)

                # Detect and analyze characters
                matched_characters = character_matcher.detect_and_match_characters(img_path)

                # Update character statistics
                character_stats['total_characters_detected'] += len(matched_characters)

                for char in matched_characters:
                    # Get primary orientation
                    orientation = max(char['orientation']['orientation'].items(),
                                      key=lambda x: x[1])[0]
                    character_stats['characters_by_orientation'][orientation] += 1

                    # Track character matches
                    if char['matches']:
                        char_name = char['matches'][0]['name']  # Best match
                        if char_name not in character_stats['character_matches']:
                            character_stats['character_matches'][char_name] = 0
                        character_stats['character_matches'][char_name] += 1

                # Store content analysis results
                analysis_result = {
                    'file_name': img_path.name,
                    'nsfw_score': nsfw_score,
                    'content_scores': content_scores,
                    'character_analysis': matched_characters
                }

                # Save analysis results
                result_file = results_dir / f"{img_path.stem}_analysis.json"
                with open(result_file, 'w') as f:
                    json.dump(analysis_result, f, indent=2)

                # Determine if image is suitable for training
                if not is_nsfw and content_scores['overall']['safe'] > 0.6:
                    # Randomly assign to train or validation set
                    dest_dir = val_dir if np.random.random() < val_split else train_dir

                    # Copy image and its analysis
                    shutil.copy2(img_path, dest_dir / img_path.name)

            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue

        # Save character database
        character_matcher.character_db.save_database()

        # Save character statistics
        with open(results_dir / "character_statistics.json", 'w') as f:
            json.dump(character_stats, f, indent=2)

        # Check if we have enough images for training
        train_images = list(train_dir.glob('*.[jp][pn][g]'))
        val_images = list(val_dir.glob('*.[jp][pn][g]'))

        if len(train_images) < Config.BATCH_SIZE or len(val_images) < Config.BATCH_SIZE:
            raise ValueError(
                f"Insufficient images for training. Found {len(train_images)} training and {len(val_images)} validation images")

        logger.info(f"Processed images: {len(train_images)} training, {len(val_images)} validation")
        logger.info(f"Total characters detected: {character_stats['total_characters_detected']}")

        # Initialize datasets with character detection
        train_dataset = RobustImageDataset(train_dir, transform=preprocessor.transform)
        val_dataset = RobustImageDataset(val_dir, transform=preprocessor.transform)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=custom_collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=custom_collate_fn
        )

        # Initialize model with character detection capabilities
        num_classes_dict = {
            category: len(classes)
            for category, classes in Config.CATEGORIES.items()
        }
        # Add character orientation classes
        num_classes_dict.update({
            'character_orientation': len(Config.ORIENTATION_DETECTION['orientation_categories']),
            'character_visibility': len(Config.ORIENTATION_DETECTION['pose_attributes'])
        })

        model = RobustImageClassifier(num_classes_dict)

        # Train model
        trainer = ModelTrainer(model, train_loader, val_loader, Config.DEVICE)
        model_state = trainer.train(Config.NUM_EPOCHS)

        # Save model and configuration
        torch.save(model_state, results_dir / "final_model.pth")

        # Save extended configuration including character detection settings
        config_data = {
            'model_name': Config.MODEL_NAME,
            'categories': Config.CATEGORIES,
            'thresholds': Config.CONFIDENCE_THRESHOLDS,
            'image_size': Config.IMG_SIZE,
            'orientation_detection': Config.ORIENTATION_DETECTION,
            'character_detection': {
                'enabled': True,
                'confidence_threshold': Config.CHARACTER_DETECTION['confidence_threshold'],
                'max_characters': Config.CHARACTER_DETECTION['max_characters'],
                'orientation_tracking': True
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(results_dir / "config.json", 'w') as f:
            json.dump(config_data, f, indent=2)

        # Save character analysis summary
        character_summary = {
            'total_images_processed': len(input_images),
            'total_characters_detected': character_stats['total_characters_detected'],
            'orientation_distribution': character_stats['characters_by_orientation'],
            'character_matches': character_stats['character_matches'],
            'processing_timestamp': datetime.now().isoformat()
        }

        with open(results_dir / "character_analysis_summary.json", 'w') as f:
            json.dump(character_summary, f, indent=2)

        logger.info(f"Training completed. Results saved to {results_dir}")
        logger.info(f"Character detection summary: {character_summary}")

    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()