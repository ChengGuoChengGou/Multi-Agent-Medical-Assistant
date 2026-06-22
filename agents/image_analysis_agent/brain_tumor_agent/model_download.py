import logging
import os

logger = logging.getLogger(__name__)

def download_model_checkpoint(gdrive_file_id, output_path):
    """
    Download model checkpoint from Google Drive if it doesn't exist.
    
    Args:
        gdrive_file_id (str): Google Drive file ID
        output_path (str): Path where model will be saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        logger.info(f"Model checkpoint already exists at {output_path}")
        return
    
    try:
        import gdown
        logger.info(f"Downloading model checkpoint to {output_path}...")
        url = f'https://drive.google.com/uc?id={gdrive_file_id}'
        gdown.download(url, output_path, quiet=False)
        logger.info("Download complete!")
    except ImportError:
        logger.warning("gdown not installed. Install with: pip install gdown")
        raise FileNotFoundError(
            f"Model not found at {output_path}. Install gdown and set BRAIN_TUMOR_MODEL_GDRIVE_ID env var."
        )
