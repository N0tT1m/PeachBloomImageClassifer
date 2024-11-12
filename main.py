# part1_core.py
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import os
import json
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List, Tuple
import cv2
import tempfile
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    # Model settings
    MODEL_NAME = "efficientnet_b0"
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enhanced categories with NSFW attributes
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
            'indoor', 'outdoor', 'school', 'city', 'nature',
            'private_room', 'bath_scene', 'bedroom'
        ],
        'emotion': ['happy', 'serious', 'surprised', 'neutral', 'seductive'],
        'clothing': [
            'casual', 'formal', 'sportswear', 'outerwear',
            'swimwear', 'lingerie', 'partial', 'minimal',
            'none'
        ],
        'clothing_coverage': [
            'full', 'moderate', 'revealing', 'minimal', 'none'
        ],
        'clothing_style': [
            'conservative', 'fashionable', 'sporty',
            'suggestive', 'revealing', 'intimate'
        ],
        'composition': [
            'portrait', 'full_body', 'group', 'candid',
            'intimate_shot', 'suggestive_angle'
        ],
        'background': [
            'indoor', 'outdoor', 'studio', 'natural',
            'bedroom', 'bathroom', 'private_setting'
        ],
        'background_context': [
            'public', 'private', 'intimate', 'suggestive_setting'
        ],
        'hair_color': ['black', 'brown', 'blonde', 'blue', 'pink', 'other'],
        'eye_color': ['brown', 'blue', 'green', 'red', 'purple', 'other'],
        'content_rating': ['safe', 'suggestive', 'questionable', 'explicit'],
        'age_rating': ['all_ages', 'teen', 'mature', 'adult'],
        'safety_tags': [
            'safe', 'suggestive', 'questionable',
            'nsfw', 'explicit', 'blocked'
        ]
    }

    # Enhanced confidence thresholds for sensitive categories
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

    # Safety settings
    SAFETY_LEVELS = {
        'safe': {
            'allowed_poses': ['standing', 'sitting', 'running', 'profile'],
            'allowed_clothing': ['casual', 'formal', 'sportswear', 'outerwear'],
            'allowed_backgrounds': ['indoor', 'outdoor', 'studio', 'natural'],
            'allowed_art_styles': ['cel_shaded', 'digital', 'watercolor', 'sketch']
        },
        'moderate': {
            'blocked_poses': ['explicit_pose'],
            'blocked_clothing': ['none'],
            'blocked_backgrounds': ['private_setting'],
            'blocked_art_styles': ['erotic_art']
        },
        'complete': {
            'block_nothing': True
        }
    }


class ImagePreprocessor:
    def __init__(self, target_size: int = Config.IMAGE_SIZE):
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


class ContentAnalyzer:
    """Enhanced content analysis with comprehensive NSFW detection"""

    def __init__(self):
        try:
            from nudenet import NudeDetector
            self.detector = NudeDetector()
            logger.info("NudeDetector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NudeDetector: {str(e)}")
            raise

    def analyze_content(self, image_path: Path) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive content analysis returning scores for multiple aspects
        """
        # Initialize default scoring system
        scores = {
            'overall': {
                'safe': 1.0,
                'suggestive': 0.0,
                'explicit': 0.0
            },
            'pose': {
                'normal': 1.0,
                'suggestive': 0.0,
                'intimate': 0.0
            },
            'clothing': {
                'covered': 1.0,
                'revealing': 0.0,
                'minimal': 0.0
            },
            'context': {
                'safe': 1.0,
                'intimate': 0.0,
                'suggestive': 0.0
            },
            'art_style': {
                'standard': 1.0,
                'suggestive': 0.0,
                'erotic': 0.0
            }
        }

        try:
            # Skip hidden files and check if file exists
            if image_path.name.startswith('.') or not image_path.exists():
                logger.warning(f"Skipping invalid or hidden file: {image_path}")
                return scores

            # Check if file is actually an image
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    logger.warning(f"Could not read image file: {image_path}")
                    return scores
            except Exception as e:
                logger.warning(f"Error reading image {image_path}: {str(e)}")
                return scores

            # Create a temporary copy of the image for processing
            with tempfile.NamedTemporaryFile(suffix=image_path.suffix) as temp_file:
                cv2.imwrite(temp_file.name, img)
                detections = self.detector.detect(temp_file.name)

            if not detections:
                return scores

            # Analyze detections for multiple aspects
            for detection in detections:
                score = detection.get('score', 0)
                label = detection.get('label', '').lower()

                # Update scores based on detection type
                self._update_scores(scores, label, score)

            # Normalize and adjust scores
            self._normalize_scores(scores)

        except Exception as e:
            logger.error(f"Error analyzing content for {image_path}: {str(e)}")
            # Return default scores in case of error
            return scores

        return scores

    def _update_scores(self, scores: Dict[str, Dict[str, float]], label: str, score: float):
        """Update various aspect scores based on detection"""
        # Update scores based on detection type
        if 'explicit' in label or 'nude' in label:
            scores['overall']['explicit'] = max(scores['overall']['explicit'], score)
            scores['overall']['safe'] *= (1 - score)
        elif 'suggestive' in label or 'intimate' in label:
            scores['overall']['suggestive'] = max(scores['overall']['suggestive'], score)
            scores['overall']['safe'] *= (1 - score * 0.7)

        # ... (rest of the existing _update_scores method remains the same)

    def _normalize_scores(self, scores: Dict[str, Dict[str, float]]):
        """Normalize all scores to ensure they sum to 1.0 within each category"""
        for category in scores:
            total = sum(scores[category].values())
            if total > 0:
                for key in scores[category]:
                    scores[category][key] /= total


class RobustImageDataset(Dataset):
    """Enhanced dataset class with comprehensive content analysis and recursive scanning"""

    def __init__(self, image_dir: str, labels_file: Optional[str] = None, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.data = []
        self._content_analyzer = None  # Initialize as None

        # Ensure image directory exists
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")

        # Load or create labels
        if labels_file and Path(labels_file).exists():
            with open(labels_file, 'r') as f:
                self.data = json.load(f)['images']
        else:
            # Recursively find all image files
            image_files = self._find_images_recursive(self.image_dir)

            if not image_files:
                raise ValueError(f"No valid images found in directory: {image_dir}")

            logger.info(f"Found {len(image_files)} images in {image_dir}")

            # Store file paths
            for img_path in image_files:
                self.data.append({
                    'file_name': str(img_path.relative_to(self.image_dir)),
                    'labels': None
                })

    def _get_content_analyzer(self):
        """Lazy initialization of ContentAnalyzer"""
        if self._content_analyzer is None:
            self._content_analyzer = ContentAnalyzer()
        return self._content_analyzer

    def _find_images_recursive(self, directory: Path) -> List[Path]:
        """Recursively find all valid images in directory and subdirectories"""
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
        """Generate comprehensive labels based on content analysis"""
        labels = {}

        # Art style
        style_scores = content_scores['art_style']
        if style_scores['erotic'] > 0.7:
            labels['art_style'] = 'erotic_art'
        elif style_scores['suggestive'] > 0.7:
            labels['art_style'] = 'suggestive_style'
        else:
            labels['art_style'] = 'cel_shaded'  # Use valid default from Config

        # Pose
        pose_scores = content_scores['pose']
        if pose_scores['intimate'] > 0.7:
            labels['character_pose'] = 'intimate_pose'
        elif pose_scores['suggestive'] > 0.7:
            labels['character_pose'] = 'suggestive_pose'
        else:
            labels['character_pose'] = 'standing'

        # Clothing
        clothing_scores = content_scores['clothing']
        if clothing_scores['minimal'] > 0.7:
            labels['clothing'] = 'minimal'
            labels['clothing_coverage'] = 'minimal'
        elif clothing_scores['revealing'] > 0.7:
            labels['clothing'] = 'revealing'
            labels['clothing_coverage'] = 'revealing'
        else:
            labels['clothing'] = 'casual'
            labels['clothing_coverage'] = 'full'

        # Background context
        context_scores = content_scores['context']
        if context_scores['intimate'] > 0.7:
            labels['background_context'] = 'intimate'
        elif context_scores['suggestive'] > 0.7:
            labels['background_context'] = 'suggestive_setting'
        else:
            labels['background_context'] = 'public'

        # Overall content rating
        overall_scores = content_scores['overall']
        if overall_scores['explicit'] > 0.7:
            labels['content_rating'] = 'explicit'
            labels['age_rating'] = 'adult'
        elif overall_scores['suggestive'] > 0.7:
            labels['content_rating'] = 'suggestive'
            labels['age_rating'] = 'mature'
        else:
            labels['content_rating'] = 'safe'
            labels['age_rating'] = 'all_ages'

        # Fill in remaining categories with defaults
        default_labels = self._get_default_labels()
        for category in Config.CATEGORIES:
            if category not in labels:
                labels[category] = default_labels[category]

        # Validate all labels
        for category, label in labels.items():
            if label not in Config.CATEGORIES[category]:
                logger.warning(f"Invalid label {label} for category {category}, using default")
                labels[category] = Config.CATEGORIES[category][0]

        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self.image_dir / item['file_name']

        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            image = torch.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        # Generate labels if not already present
        if item['labels'] is None:
            try:
                analyzer = self._get_content_analyzer()
                content_scores = analyzer.analyze_content(image_path)
                item['labels'] = self._generate_labels(content_scores)
            except Exception as e:
                logger.error(f"Error analyzing content for {image_path}: {str(e)}")
                item['labels'] = self._get_default_labels()

        # Convert labels to tensors
        tensor_labels = {}
        for category, label in item['labels'].items():
            if category in Config.CATEGORIES:
                try:
                    # Ensure label is valid
                    if label not in Config.CATEGORIES[category]:
                        logger.warning(f"Invalid label {label} for category {category}, using default")
                        label = Config.CATEGORIES[category][0]

                    # Convert label to index
                    label_idx = Config.CATEGORIES[category].index(label)
                    tensor_labels[category] = torch.tensor(label_idx, dtype=torch.long)
                except Exception as e:
                    logger.error(f"Error converting label {label} for category {category}: {str(e)}")
                    tensor_labels[category] = torch.tensor(0, dtype=torch.long)

        return {
            'image': image,
            'file_name': item['file_name'],
            'labels': tensor_labels
        }

    def _get_default_labels(self):
        """Return default labels for all categories using valid values from Config"""
        return {
            'art_style': Config.CATEGORIES['art_style'][0],  # First value is default
            'character_pose': Config.CATEGORIES['character_pose'][0],
            'scene_type': Config.CATEGORIES['scene_type'][0],
            'emotion': Config.CATEGORIES['emotion'][0],
            'clothing': Config.CATEGORIES['clothing'][0],
            'clothing_coverage': Config.CATEGORIES['clothing_coverage'][0],
            'clothing_style': Config.CATEGORIES['clothing_style'][0],
            'composition': Config.CATEGORIES['composition'][0],
            'background': Config.CATEGORIES['background'][0],
            'background_context': Config.CATEGORIES['background_context'][0],
            'hair_color': Config.CATEGORIES['hair_color'][0],
            'eye_color': Config.CATEGORIES['eye_color'][0],
            'content_rating': Config.CATEGORIES['content_rating'][0],
            'age_rating': Config.CATEGORIES['age_rating'][0],
            'safety_tags': Config.CATEGORIES['safety_tags'][0]
        }

    def save_labels(self, output_file: str):
        """Save generated labels to file"""
        with open(output_file, 'w') as f:
            json.dump({'images': self.data}, f, indent=2)

class RobustImageClassifier(nn.Module):
    def __init__(self, num_classes_dict):
        super().__init__()
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        feat_dim = self.backbone.num_features

        # Enhanced classifier heads with attention for sensitive categories
        self.classifiers = nn.ModuleDict()
        for category, num_classes in num_classes_dict.items():
            if category in ['content_rating', 'nsfw_attributes', 'age_rating',
                            'clothing', 'character_pose', 'art_style']:
                # Complex classifier for sensitive categories
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
                # Standard classifier for other categories
                self.classifiers[category] = nn.Sequential(
                    nn.BatchNorm1d(feat_dim),
                    nn.Linear(feat_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )

        # Initialize weights
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
        """Get probability distributions for all categories"""
        outputs = self.forward(x)
        return {
            category: torch.softmax(output, dim=1)
            for category, output in outputs.items()
        }

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # Loss functions with label smoothing for sensitive categories
        self.criterion = {}
        for category in Config.CATEGORIES.keys():
            if category in ['content_rating', 'nsfw_attributes', 'age_rating',
                            'clothing', 'character_pose', 'art_style']:
                self.criterion[category] = nn.CrossEntropyLoss(label_smoothing=0.1)
            else:
                self.criterion[category] = nn.CrossEntropyLoss()

    def train(self, num_epochs):
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch()

            # Validation phase
            val_loss, val_metrics = self._validate()

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                torch.save(best_model_state, 'best_model.pth')

            # Log metrics
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

            # Calculate loss for each category
            loss = sum(
                self.criterion[cat](outputs[cat], labels[cat])
                for cat in outputs.keys()
            )

            loss.backward()

            # Gradient clipping
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

                # Calculate loss for each category
                loss = sum(
                    self.criterion[cat](outputs[cat], labels[cat])
                    for cat in outputs.keys()
                )
                total_loss += loss.item()

                # Calculate accuracy for each category
                for category in outputs:
                    _, preds = torch.max(outputs[category], 1)
                    metrics[category]['correct'] += (preds == labels[category]).sum().item()
                    metrics[category]['total'] += labels[category].size(0)

        # Calculate final metrics
        val_loss = total_loss / len(self.val_loader)
        accuracies = {
            category: metrics[category]['correct'] / metrics[category]['total']
            for category in metrics
        }

        return val_loss, accuracies

    def _log_metrics(self, epoch, train_loss, val_loss, val_metrics):
        logger.info(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")

        # Log metrics by category groups
        sensitive_categories = ['content_rating', 'nsfw_attributes', 'age_rating',
                                'clothing', 'character_pose', 'art_style']

        logger.info("\nSensitive Categories:")
        for category in sensitive_categories:
            if category in val_metrics:
                logger.info(f"{category}: {val_metrics[category]:.4f}")

        logger.info("\nOther Categories:")
        for category, accuracy in val_metrics.items():
            if category not in sensitive_categories:
                logger.info(f"{category}: {accuracy:.4f}")


def custom_collate_fn(batch):
    """Custom collate function to handle None values and ensure proper batching"""
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

            # Collect labels
            for category, tensor in item['labels'].items():
                if tensor is not None:
                    collated['labels'][category].append(tensor)

    # Stack tensors if we have any valid items
    if collated['image']:
        collated['image'] = torch.stack(collated['image'])

        # Stack label tensors
        for category in collated['labels']:
            if collated['labels'][category]:
                collated['labels'][category] = torch.stack(collated['labels'][category])

    return collated


class ImageClassificationPipeline:
    def __init__(self, model_path: Optional[str] = None):
        self.preprocessor = ImagePreprocessor()
        self.content_analyzer = ContentAnalyzer()

        # Initialize model
        num_classes_dict = {
            category: len(classes)
            for category, classes in Config.CATEGORIES.items()
        }
        self.model = RobustImageClassifier(num_classes_dict)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            logger.info(f"Loaded model from {model_path}")

        self.model = self.model.to(Config.DEVICE)
        self.model.eval()

    def process_image(self, image_path: str) -> Dict[str, dict]:
        """Process a single image with comprehensive analysis"""
        # Analyze content
        content_scores = self.content_analyzer.analyze_content(Path(image_path))

        # Process image
        image_tensor = self.preprocessor.preprocess_image(image_path)
        if image_tensor is None:
            return {
                'error': 'Failed to process image',
                'content_scores': content_scores
            }

        image_tensor = image_tensor.unsqueeze(0).to(Config.DEVICE)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = self.model.predict_proba(image_tensor)

        predictions = {
            'content_scores': content_scores,
            'classifications': {},
            'probabilities': {}
        }

        # Process outputs with category-specific thresholds
        for category, output in outputs.items():
            probs = probabilities[category][0]  # Get probabilities for first (only) image
            conf, pred = torch.max(probs, dim=0)

            threshold = Config.CONFIDENCE_THRESHOLDS.get(
                category, Config.CONFIDENCE_THRESHOLDS['default']
            )

            if conf.item() >= threshold:
                predictions['classifications'][category] = {
                    'label': Config.CATEGORIES[category][pred.item()],
                    'confidence': conf.item()
                }

                # Store top-3 predictions with probabilities
                top3_values, top3_indices = torch.topk(probs, min(3, len(probs)))
                predictions['probabilities'][category] = [
                    {
                        'label': Config.CATEGORIES[category][idx.item()],
                        'probability': prob.item()
                    }
                    for prob, idx in zip(top3_values, top3_indices)
                ]

        return predictions

    def process_directory(self, image_dir: str, output_file: str):
        """Process all images in a directory with comprehensive analysis"""
        dataset = RobustImageDataset(
            image_dir,
            transform=self.preprocessor.transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            num_workers=0,  # Single process to avoid pickling issues
            shuffle=False,
            collate_fn=custom_collate_fn  # Use custom collate function
        )

        results = {
            "images": [],
            "statistics": {
                "total": 0,
                "by_category": {
                    category: {label: 0 for label in labels}
                    for category, labels in Config.CATEGORIES.items()
                }
            }
        }

        for batch in tqdm(dataloader, desc="Processing images"):
            if not batch or 'image' not in batch:
                continue

            images = batch['image'].to(Config.DEVICE)
            filenames = batch['file_name']

            with torch.no_grad():
                outputs = self.model(images)
                probabilities = self.model.predict_proba(images)

            for idx, filename in enumerate(filenames):
                predictions = {
                    'classifications': {},
                    'probabilities': {}
                }

                for category, output in outputs.items():
                    probs = probabilities[category][idx]
                    conf, pred = torch.max(probs, dim=0)

                    threshold = Config.CONFIDENCE_THRESHOLDS.get(
                        category, Config.CONFIDENCE_THRESHOLDS['default']
                    )

                    if conf.item() >= threshold:
                        label = Config.CATEGORIES[category][pred.item()]
                        predictions['classifications'][category] = {
                            'label': label,
                            'confidence': conf.item()
                        }

                        results['statistics']['by_category'][category][label] += 1

                        top3_values, top3_indices = torch.topk(probs, min(3, len(probs)))
                        predictions['probabilities'][category] = [
                            {
                                'label': Config.CATEGORIES[category][idx.item()],
                                'probability': prob.item()
                            }
                            for prob, idx in zip(top3_values, top3_indices)
                        ]

                results["images"].append({
                    "file_name": filename,
                    "predictions": predictions
                })

        results["statistics"]["total"] = len(results["images"])

        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def train_model(self, train_dir: str, val_dir: str, num_epochs: int = Config.NUM_EPOCHS):
        """Train or fine-tune the model"""
        logger.info(f"Starting model training with {num_epochs} epochs")
        logger.info(f"Training directory: {train_dir}")
        logger.info(f"Validation directory: {val_dir}")

        try:
            # Create datasets
            train_dataset = RobustImageDataset(
                train_dir,
                transform=self.preprocessor.transform
            )
            logger.info(f"Created training dataset with {len(train_dataset)} images")

            val_dataset = RobustImageDataset(
                val_dir,
                transform=self.preprocessor.transform
            )
            logger.info(f"Created validation dataset with {len(val_dataset)} images")

            if len(train_dataset) == 0:
                raise ValueError(f"No training images found in {train_dir}")
            if len(val_dataset) == 0:
                raise ValueError(f"No validation images found in {val_dir}")

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=0,  # Disable multiprocessing to avoid pickling issues
                pin_memory=True if torch.cuda.is_available() else False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=0,  # Disable multiprocessing to avoid pickling issues
                pin_memory=True if torch.cuda.is_available() else False
            )

            # Initialize trainer
            trainer = ModelTrainer(
                self.model,
                train_loader,
                val_loader,
                Config.DEVICE
            )

            # Train model
            best_model_state = trainer.train(num_epochs)

            # Load best model
            self.model.load_state_dict(best_model_state)
            self.model.eval()

            return self.model

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise


def setup_logging(log_file: str = 'image_classifier.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def create_results_directory(base_dir: str = 'results') -> Path:
    """Create timestamped results directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(base_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_model_info(model, save_dir: Path):
    """Save model architecture and configuration"""
    info = {
        'model_name': Config.MODEL_NAME,
        'categories': Config.CATEGORIES,
        'thresholds': Config.CONFIDENCE_THRESHOLDS,
        'image_size': Config.IMAGE_SIZE,
        'timestamp': datetime.now().isoformat()
    }

    with open(save_dir / 'model_info.json', 'w') as f:
        json.dump(info, f, indent=2)


def generate_report(results: Dict, save_path: Path):
    """Generate detailed analysis report without relying on content_scores"""
    report = {
        'summary': {
            'total_images': results['statistics']['total'],
            'timestamp': datetime.now().isoformat()
        },
        'category_statistics': {},
        'classification_distribution': {}
    }

    # Calculate statistics for each category
    for category, counts in results['statistics']['by_category'].items():
        total = sum(counts.values())
        if total > 0:
            report['category_statistics'][category] = {
                'distribution': {
                    label: count / total
                    for label, count in counts.items()
                },
                'most_common': max(counts.items(), key=lambda x: x[1])[0]
            }

    # Analyze classification distributions
    classification_counts = {}
    confidence_sums = {}
    total_images = len(results['images'])

    for image in results['images']:
        for category, pred in image['predictions']['classifications'].items():
            if category not in classification_counts:
                classification_counts[category] = {}
                confidence_sums[category] = {}

            label = pred['label']
            confidence = pred['confidence']

            # Update count
            classification_counts[category][label] = classification_counts[category].get(label, 0) + 1

            # Update confidence sum
            if label not in confidence_sums[category]:
                confidence_sums[category][label] = []
            confidence_sums[category][label].append(confidence)

    # Calculate distributions and average confidences
    for category in classification_counts:
        report['classification_distribution'][category] = {
            'label_distribution': {
                label: count / total_images
                for label, count in classification_counts[category].items()
            },
            'average_confidence': {
                label: sum(confidences) / len(confidences)
                for label, confidences in confidence_sums[category].items()
            }
        }

    # Add top predicted categories
    report['summary']['top_predictions'] = {
        category: max(dist['label_distribution'].items(), key=lambda x: x[1])[0]
        for category, dist in report['classification_distribution'].items()
    }

    # Save report
    with open(save_path / 'analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    return report


def visualize_results(results: Dict, save_dir: Path):
    """Generate visualizations of the results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        # Create visualizations directory
        vis_dir = save_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

        # Plot category distributions
        for category, counts in results['statistics']['by_category'].items():
            plt.figure(figsize=(10, 6))
            plt.bar(counts.keys(), counts.values())
            plt.title(f'Distribution of {category}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(vis_dir / f'{category}_distribution.png')
            plt.close()

        # Plot classification confidence distributions
        for image in results['images']:
            classifications = image['predictions']['classifications']
            for category, pred in classifications.items():
                if not hasattr(visualize_results, 'confidence_data'):
                    visualize_results.confidence_data = {}
                if category not in visualize_results.confidence_data:
                    visualize_results.confidence_data[category] = {
                        'labels': [],
                        'confidences': []
                    }

                visualize_results.confidence_data[category]['labels'].append(pred['label'])
                visualize_results.confidence_data[category]['confidences'].append(pred['confidence'])

        # Create confidence distribution plots
        for category, data in getattr(visualize_results, 'confidence_data', {}).items():
            plt.figure(figsize=(10, 6))
            df = pd.DataFrame({
                'Label': data['labels'],
                'Confidence': data['confidences']
            })
            sns.boxplot(x='Label', y='Confidence', data=df)
            plt.title(f'{category} Confidence Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(vis_dir / f'{category}_confidence.png')
            plt.close()

    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")

def example_single_image():
    """Example of processing a single image"""
    pipeline = ImageClassificationPipeline("best_model.pth")

    image_path = "path/to/image.jpg"
    predictions = pipeline.process_image(image_path)

    print("\nContent Scores:")
    for aspect, scores in predictions['content_scores'].items():
        print(f"\n{aspect}:")
        for label, score in scores.items():
            print(f"  {label}: {score:.3f}")

    print("\nClassifications:")
    for category, pred in predictions['classifications'].items():
        print(f"\n{category}:")
        print(f"  Label: {pred['label']}")
        print(f"  Confidence: {pred['confidence']:.3f}")

        print("  Top 3 probabilities:")
        for p in predictions['probabilities'][category]:
            print(f"    {p['label']}: {p['probability']:.3f}")


def example_directory_processing():
    """Example of processing a directory of images"""
    # Setup
    setup_logging()
    results_dir = create_results_directory()

    # Initialize pipeline
    pipeline = ImageClassificationPipeline("best_model.pth")

    # Process directory
    input_dir = "./hentai"
    results = pipeline.process_directory(
        input_dir,
        results_dir / "predictions.json"
    )

    # Generate report and visualizations
    report = generate_report(results, results_dir)
    visualize_results(results, results_dir)

    # Print summary
    print("\nProcessing Summary:")
    print(f"Total images processed: {report['summary']['total_images']}")

    print("\nCategory Statistics:")
    for category, stats in report['category_statistics'].items():
        print(f"\n{category}:")
        print(f"Most common: {stats['most_common']}")
        print("Distribution:")
        for label, percentage in stats['distribution'].items():
            print(f"  {label}: {percentage:.1%}")


def example_model_training():
    """Example of training the model"""
    try:
        # Setup
        setup_logging()
        results_dir = create_results_directory()
        logger.info(f"Created results directory at {results_dir}")

        # Initialize pipeline
        pipeline = ImageClassificationPipeline()
        logger.info("Initialized classification pipeline")

        # Check directories
        train_dir = "./training-hentai"
        val_dir = "./validated-hentai"

        # Verify directories exist
        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory does not exist: {train_dir}")
        if not os.path.exists(val_dir):
            raise ValueError(f"Validation directory does not exist: {val_dir}")

        # Count images in directories
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
        train_images = [f for f in Path(train_dir).glob('*.*')
                        if f.suffix.lower() in valid_extensions and not f.name.startswith('.')]
        val_images = [f for f in Path(val_dir).glob('*.*')
                      if f.suffix.lower() in valid_extensions and not f.name.startswith('.')]

        logger.info(f"Found {len(train_images)} training images and {len(val_images)} validation images")

        if not train_images:
            raise ValueError(f"No valid images found in training directory: {train_dir}")
        if not val_images:
            raise ValueError(f"No valid images found in validation directory: {val_dir}")

        # Train model
        logger.info("Starting model training...")
        trained_model = pipeline.train_model(
            train_dir=train_dir,
            val_dir=val_dir,
            num_epochs=Config.NUM_EPOCHS
        )

        # Save model and info
        model_path = results_dir / "final_model.pth"
        torch.save(trained_model.state_dict(), model_path)
        logger.info(f"Saved trained model to {model_path}")

        save_model_info(trained_model, results_dir)
        logger.info(f"Saved model info to {results_dir}")

        print(f"\nModel training completed successfully!")
        print(f"Results saved to: {results_dir}")

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    # print("Processing single image...")
    # example_single_image()

    print("\nProcessing directory...")
    example_directory_processing()

    print("\nTraining model...")
    example_model_training()