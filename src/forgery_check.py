"""
Forgery Detection Module for Driver License Documents
Combines OCR-based checks with image forensics
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Try imports - gracefully handle if not installed
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


@dataclass
class ForgeryResult:
    """Result of forgery detection analysis."""
    is_forged: bool
    confidence: float
    ocr_text: str
    issues: List[str]
    field_scores: Dict[str, float]


class ForgeryDetector:
    """Detect forged documents using OCR and image analysis."""
    
    # Expected patterns for UAE license (example)
    LICENSE_PATTERNS = {
        'license_no': r'UAE-[A-Z]{3}-\d{6}',
        'date': r'\d{4}-\d{2}-\d{2}',
        'name': r'[A-Za-z\s]{2,50}',
    }
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize detector with optional Tesseract path."""
        if tesseract_path and TESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using OCR."""
        if not TESSERACT_AVAILABLE:
            return "[Tesseract not installed]"
        
        if not CV2_AVAILABLE:
            return "[OpenCV not installed]"
        
        img = cv2.imread(image_path)
        if img is None:
            return "[Could not read image]"
        
        # Preprocess: convert to grayscale and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        text = pytesseract.image_to_string(gray)
        return text
    
    def validate_license_number(self, text: str) -> Tuple[bool, float]:
        """Check if license number matches expected format."""
        pattern = self.LICENSE_PATTERNS['license_no']
        matches = re.findall(pattern, text)
        
        if matches:
            return True, 1.0
        
        # Check for partial matches or suspicious patterns
        partial_pattern = r'UAE-[A-Z0-9]{3}-[A-Z0-9]{6}'
        partial_matches = re.findall(partial_pattern, text)
        
        if partial_matches:
            return False, 0.5  # Suspicious - might be tampered
        
        return False, 0.0  # No license number found
    
    def validate_date(self, text: str) -> Tuple[bool, float]:
        """Check if dates are valid and not expired."""
        pattern = self.LICENSE_PATTERNS['date']
        dates = re.findall(pattern, text)
        
        if not dates:
            return False, 0.0
        
        valid_count = 0
        for date_str in dates:
            try:
                year, month, day = map(int, date_str.split('-'))
                if 1 <= month <= 12 and 1 <= day <= 31 and 1950 <= year <= 2030:
                    valid_count += 1
            except:
                pass
        
        score = valid_count / len(dates) if dates else 0
        return score > 0.5, score
    
    def check_image_quality(self, image_path: str) -> Tuple[bool, float, List[str]]:
        """Analyze image for quality issues that might indicate forgery."""
        issues = []
        
        if not CV2_AVAILABLE:
            return True, 0.5, ["OpenCV not installed - skipped image checks"]
        
        img = cv2.imread(image_path)
        if img is None:
            return False, 0.0, ["Could not read image"]
        
        # Check resolution
        height, width = img.shape[:2]
        if width < 300 or height < 200:
            issues.append("Low resolution image")
        
        # Check for blur using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            issues.append("Image appears blurry")
        
        # Check for edge anomalies (sudden changes might indicate paste)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        if edge_density > 0.3:
            issues.append("Unusual edge density - possible manipulation")
        
        # Calculate quality score
        score = 1.0
        score -= len(issues) * 0.2
        score = max(0, score)
        
        return len(issues) == 0, score, issues
    
    def analyze(self, image_path: str) -> ForgeryResult:
        """Run full forgery detection analysis."""
        issues = []
        field_scores = {}
        
        # Extract text
        ocr_text = self.extract_text(image_path)
        
        # Validate license number
        license_valid, license_score = self.validate_license_number(ocr_text)
        field_scores['license_number'] = license_score
        if not license_valid:
            issues.append("Invalid or missing license number format")
        
        # Validate dates
        dates_valid, date_score = self.validate_date(ocr_text)
        field_scores['dates'] = date_score
        if not dates_valid:
            issues.append("Invalid or suspicious date format")
        
        # Image quality checks
        quality_ok, quality_score, quality_issues = self.check_image_quality(image_path)
        field_scores['image_quality'] = quality_score
        issues.extend(quality_issues)
        
        # Calculate overall forgery score
        avg_score = sum(field_scores.values()) / len(field_scores)
        confidence = 1 - avg_score  # Higher issues = higher forgery confidence
        
        is_forged = confidence > 0.5 or len(issues) >= 2
        
        return ForgeryResult(
            is_forged=is_forged,
            confidence=confidence,
            ocr_text=ocr_text,
            issues=issues,
            field_scores=field_scores
        )


def check_document(image_path: str) -> Dict:
    """Convenience function to check a single document."""
    detector = ForgeryDetector()
    result = detector.analyze(image_path)
    
    return {
        'is_forged': result.is_forged,
        'confidence': result.confidence,
        'issues': result.issues,
        'ocr_text': result.ocr_text[:500] if result.ocr_text else "",
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python forgery_check.py <image_path>")
        sys.exit(1)
    
    result = check_document(sys.argv[1])
    
    print("\n=== FORGERY DETECTION RESULT ===")
    print(f"Forged: {'YES' if result['is_forged'] else 'NO'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nIssues Found: {len(result['issues'])}")
    for issue in result['issues']:
        print(f"  - {issue}")
    print(f"\nExtracted Text Preview:\n{result['ocr_text'][:300]}...")
