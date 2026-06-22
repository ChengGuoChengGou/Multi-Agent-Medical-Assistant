import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BrainTumorClassification:
    """
    Brain tumor classification using EfficientNet-B0.
    Classes: glioma, meningioma, pituitary, no_tumor

    Follows the same pattern as ChestXRayClassification but uses
    EfficientNet-B0 (2019, Tan & Le) for better accuracy/efficiency.
    Falls back to ResNet50 if EfficientNet is unavailable.
    """

    CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]

    def __init__(self, model_path, device=None, num_classes=4):
        self.device = device or DEVICE
        self.num_classes = num_classes
        self.model_path = model_path

        logger.info(f"BrainTumorClassification using device: {self.device}")

        # Try to download model if not exists
        self._ensure_model()

        # Build and load model
        self.model = self._build_model()
        self._load_model_weights()
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing (ImageNet normalization)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _ensure_model(self):
        """Download model checkpoint if not present."""
        if os.path.exists(self.model_path):
            return

        gdrive_id = os.environ.get("BRAIN_TUMOR_MODEL_GDRIVE_ID")
        if gdrive_id:
            from .model_download import download_model_checkpoint

            download_model_checkpoint(gdrive_id, self.model_path)
        else:
            logger.warning(
                f"Brain tumor model not found at {self.model_path} and "
                f"BRAIN_TUMOR_MODEL_GDRIVE_ID not set. "
                f"Prediction will use randomly initialized weights."
            )

    def _build_model(self):
        """
        Build EfficientNet-B0 with custom classifier head.
        Falls back to ResNet50 if efficientnet not available.
        """
        try:
            # EfficientNet-B0: lightweight, accurate for medical imaging
            model = models.efficientnet_b0(weights=None)
            # Replace classifier head
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features, self.num_classes))
            logger.info("Built EfficientNet-B0 model for brain tumor classification")
            return model
        except (AttributeError, Exception) as e:
            logger.warning(f"EfficientNet unavailable ({e}), falling back to ResNet50")
            model = models.resnet50(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)
            logger.info("Built ResNet50 model for brain tumor classification")
            return model

    def _load_model_weights(self):
        """Load pre-trained model weights from checkpoint."""
        if not os.path.exists(self.model_path):
            logger.warning(f"No checkpoint at {self.model_path}, using random weights")
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # Handle both raw state_dict and wrapped checkpoint formats
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present (DataParallel)
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(cleaned, strict=False)
            logger.info(f"Model weights loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            logger.warning("Using randomly initialized weights")

    def predict(self, image_path):
        """
        Predict brain tumor type from MRI image.

        Args:
            image_path: Path to the MRI image file

        Returns:
            dict: {
                'prediction': str (class name),
                'confidence': float,
                'all_probabilities': dict (class -> probability)
            }
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)

                pred_class = self.CLASS_NAMES[predicted_idx.item()]
                conf_value = confidence.item()

                # All class probabilities
                all_probs = {
                    self.CLASS_NAMES[i]: round(probabilities[i].item(), 4) for i in range(len(self.CLASS_NAMES))
                }

            result = {"prediction": pred_class, "confidence": round(conf_value, 4), "all_probabilities": all_probs}

            logger.info(f"Brain tumor prediction: {pred_class} (confidence: {conf_value:.2%})")
            return result

        except Exception as e:
            logger.error(f"Error during brain tumor prediction: {str(e)}")
            return {"prediction": "error", "confidence": 0.0, "all_probabilities": {}, "error": str(e)}
