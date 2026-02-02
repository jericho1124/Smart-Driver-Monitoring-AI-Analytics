"""
Synthetic Data Generator for Smart Driver Monitoring
Generates telematics, ratings, feedback, and sample license images
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_telematics_data(n_records: int = 1000, n_drivers: int = 50) -> pd.DataFrame:
    """Generate synthetic telematics data."""
    
    drivers = [f"D{d:03d}" for d in range(n_drivers)]
    trips = [f"T{t:05d}" for t in range(n_records)]
    
    data = {
        'trip_id': trips,
        'driver_id': np.random.choice(drivers, n_records),
        'timestamp': pd.date_range('2024-01-01', periods=n_records, freq='30min'),
        'speed': np.clip(np.random.normal(45, 15, n_records), 0, 140),
        'throttle': np.clip(np.random.normal(0.5, 0.2, n_records), 0, 1),
        'brake': np.clip(np.random.normal(0.2, 0.15, n_records), 0, 1),
        'steering_angle': np.random.normal(0, 15, n_records),
        'gps_lat': np.random.uniform(24.0, 26.0, n_records),
        'gps_lon': np.random.uniform(54.0, 56.0, n_records),
        'accel_x': np.random.normal(0, 0.3, n_records),
        'accel_y': np.random.normal(0, 0.25, n_records),
        'accel_z': np.random.normal(9.8, 0.1, n_records),
        'trip_duration_sec': np.random.randint(300, 3600, n_records),
        'distance_km': np.random.uniform(2, 50, n_records),
    }
    
    df = pd.DataFrame(data)
    
    # Generate event codes based on behavior
    df['hard_brake'] = ((df['brake'] > 0.4) | (np.random.rand(n_records) < 0.08)).astype(int)
    df['overspeed'] = (df['speed'] > 80).astype(int)
    df['harsh_turn'] = (np.abs(df['steering_angle']) > 25).astype(int)
    
    return df


def generate_ratings_data(telematics_df: pd.DataFrame) -> pd.DataFrame:
    """Generate driver ratings based on telematics behavior."""
    
    # Aggregate by trip
    agg = telematics_df.groupby(['trip_id', 'driver_id']).agg({
        'speed': 'mean',
        'hard_brake': 'sum',
        'overspeed': 'sum',
        'harsh_turn': 'sum'
    }).reset_index()
    
    # Calculate rating (influenced by driving behavior)
    base_rating = 4.5
    speed_penalty = (agg['speed'] - 40) / 100  # Faster = lower rating
    brake_penalty = agg['hard_brake'] * 0.3
    overspeed_penalty = agg['overspeed'] * 0.5
    harsh_turn_penalty = agg['harsh_turn'] * 0.2
    
    ratings = (base_rating 
               - speed_penalty 
               - brake_penalty 
               - overspeed_penalty 
               - harsh_turn_penalty
               + np.random.normal(0, 0.3, len(agg)))
    
    agg['rating'] = np.clip(ratings, 1, 5).round().astype(int)
    agg['complaints_count'] = np.where(agg['rating'] <= 2, 
                                        np.random.randint(1, 4, len(agg)), 
                                        0)
    
    return agg[['trip_id', 'driver_id', 'rating', 'complaints_count']]


def generate_feedback_data(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Generate passenger feedback text based on ratings."""
    
    positive_feedback = [
        "Great driver, very professional and courteous.",
        "Smooth ride, arrived on time. Would recommend!",
        "Excellent service, driver was very helpful.",
        "Safe driving, comfortable journey. Thank you!",
        "Very punctual and polite driver. 5 stars!",
        "Amazing experience, car was clean and driver friendly.",
        "Best ride I've had, driver knew the routes well.",
        "Professional driver, great music and AC was perfect.",
    ]
    
    neutral_feedback = [
        "Ride was okay, nothing special.",
        "Average experience, got to destination fine.",
        "Driver was quiet but professional.",
        "Normal ride, no complaints.",
        "Standard service, car was clean.",
        "Reached on time, ride was fine.",
    ]
    
    negative_feedback = [
        "Driver was rude and drove too fast.",
        "Very rough ride, too many sudden brakes.",
        "Driver was on phone during the trip, felt unsafe.",
        "Car was dirty and driver was late.",
        "Dangerous driving, almost had an accident.",
        "Driver took longer route, felt cheated.",
        "Very uncomfortable ride, steering was jerky.",
        "Driver was aggressive with other cars on road.",
    ]
    
    feedbacks = []
    for _, row in ratings_df.iterrows():
        if row['rating'] >= 4:
            text = random.choice(positive_feedback)
        elif row['rating'] == 3:
            text = random.choice(neutral_feedback)
        else:
            text = random.choice(negative_feedback)
        
        feedbacks.append({
            'trip_id': row['trip_id'],
            'feedback_text': text,
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=random.randint(0, 720))
        })
    
    return pd.DataFrame(feedbacks)


def generate_sample_license_images(output_dir: Path, n_genuine: int = 5, n_forged: int = 5):
    """Generate sample license images (placeholder - creates text files with info)."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Since we can't easily generate real images, create info files
    # In a real scenario, you'd use actual license images
    
    genuine_info = """
LICENSE IMAGE INFO (GENUINE)
===========================
Name: John Ahmed Smith
License No: UAE-DXB-123456
DOB: 1985-03-15
Expiry: 2026-03-15
Category: Light Vehicle

Note: This is a placeholder. In the real project, 
use actual scanned license images.
"""
    
    forged_info = """
LICENSE IMAGE INFO (FORGED)
===========================
Name: J0hn Ahmed Sm1th  (NOTE: Contains digit substitutions)
License No: UAE-DXB-XXXXXX (NOTE: Invalid format)
DOB: 1985-13-45 (NOTE: Invalid date)
Expiry: 2020-03-15 (NOTE: Expired)
Category: Heavy Vehicle (NOTE: Mismatch)

FORGERY INDICATORS:
- Inconsistent fonts
- Blurry edges around photo
- Missing security features

Note: This is a placeholder. In the real project,
create modified images using OpenCV.
"""
    
    for i in range(n_genuine):
        with open(output_dir / f"genuine_{i+1}.txt", 'w') as f:
            f.write(genuine_info.replace("John", f"Person{i+1}"))
    
    for i in range(n_forged):
        with open(output_dir / f"forged_{i+1}.txt", 'w') as f:
            f.write(forged_info.replace("J0hn", f"Pers0n{i+1}"))
    
    print(f"Created {n_genuine} genuine and {n_forged} forged license placeholders in {output_dir}")


def main():
    """Generate all synthetic datasets."""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("Generating telematics data...")
    telematics_df = generate_telematics_data(n_records=1000, n_drivers=50)
    telematics_df.to_csv(data_dir / "telematics.csv", index=False)
    print(f"  Saved {len(telematics_df)} records to data/telematics.csv")
    
    print("Generating ratings data...")
    ratings_df = generate_ratings_data(telematics_df)
    ratings_df.to_csv(data_dir / "ratings.csv", index=False)
    print(f"  Saved {len(ratings_df)} records to data/ratings.csv")
    
    print("Generating feedback data...")
    feedback_df = generate_feedback_data(ratings_df)
    feedback_df.to_csv(data_dir / "feedback.csv", index=False)
    print(f"  Saved {len(feedback_df)} records to data/feedback.csv")
    
    print("Generating sample license placeholders...")
    generate_sample_license_images(data_dir / "licenses", n_genuine=5, n_forged=5)
    
    print("\n[OK] All synthetic data generated successfully!")


if __name__ == "__main__":
    main()
