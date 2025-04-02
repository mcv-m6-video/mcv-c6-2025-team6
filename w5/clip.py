import os
import json
from moviepy import VideoFileClip
from tqdm import tqdm

data_root = "C:/Users/laila/CVMasterActionRecognitionSpotting/SoccerNet"
input_roots = {
    #"train": os.path.join(data_root, "train"),
    "val": os.path.join(data_root, "valid"),
    "test": os.path.join(data_root, "test")
}
output_roots = {
    #"train": os.path.join(data_root, "clips/train_root"),
    "val": os.path.join(data_root, "clips/val_root"),
    "test": os.path.join(data_root, "clips/test_root")
}
class_file = os.path.join(data_root, "class.txt")
clip_duration = 5 
video_extensions = (".mp4", ".mkv")  

def load_classes():
    """Load classes from file, excluding background"""
    with open(class_file) as f:
        return [line.strip() for line in f if line.strip() != "background"]

def create_output_dirs(classes):
    """Create output directories for all splits and classes"""
    for split, output_root in output_roots.items():
        os.makedirs(output_root, exist_ok=True)  
        for cls in classes:
            os.makedirs(os.path.join(output_root, cls), exist_ok=True)

def parse_game_time(game_time):
    """Convert timestamp to seconds (e.g., "1 - 12:34" → 744.0)"""
    half, time = game_time.split(" - ")
    minutes, seconds = map(int, time.split(":"))
    return (int(half) - 1) * 45 * 60 + minutes * 60 + seconds

def extract_clip(video, start_time, duration, output_path):
    """Extract a clip from video and save it"""
    try:
        end_time = min(start_time + duration, video.duration)
        if start_time >= end_time:  
            return False
            
        clip = video.with_subclip(start_time, end_time)
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio=False,
            threads=4,  
            preset='fast',  
            ffmpeg_params=['-crf', '23']  
        )
        clip.close()
        return True
    except Exception as e:
        print(f"❌ Failed to extract {os.path.basename(output_path)}: {str(e)}")
        return False

def process_split(split_name):
    """Process all matches in a specific split (train, val, or test)"""
    print(f"\nProcessing {split_name} split...")
    input_root = input_roots[split_name]
    
    if not os.path.exists(input_root):
        print(f"⚠️ Directory not found: {input_root}")
        return
        
    output_root = output_roots[split_name]
    
    for league in os.listdir(input_root):
        league_dir = os.path.join(input_root, league)
        if not os.path.isdir(league_dir):
            continue
            
        for year in os.listdir(league_dir):
            year_dir = os.path.join(league_dir, year)
            if not os.path.isdir(year_dir):
                continue
                
            for match_folder in os.listdir(year_dir):
                match_dir = os.path.join(year_dir, match_folder)
                if not os.path.isdir(match_dir):
                    continue
                
                json_file = None
                video_file = None
                
                for f in os.listdir(match_dir):
                    file_path = os.path.join(match_dir, f)
                    if f.endswith(".json"):
                        json_file = file_path
                    elif f.lower().endswith(video_extensions):
                        video_file = file_path
                
                if not json_file:
                    print(f"⚠️ JSON not found in {league}/{year}/{match_folder}")
                    continue
                
                if not video_file:
                    print(f"⚠️ Video not found for {league}/{year}/{match_folder}")
                    continue
                
                try:
                    with open(json_file) as f:
                        labels = json.load(f)
                except Exception as e:
                    print(f"⚠️ Failed to load JSON {json_file}: {str(e)}")
                    continue
                
                try:
                    video = VideoFileClip(video_file)
                except Exception as e:
                    print(f"⚠️ Failed to load video {video_file}: {str(e)}")
                    continue
                
                annotations = labels.get("annotations", [])
                for ann in tqdm(annotations, desc=f"{match_folder}"):
                    cls = ann.get("label")
                    if cls not in classes:
                        continue
                    
                    try:
                        game_time = ann.get("gameTime")
                        if not game_time:
                            continue
                            
                        start_time = parse_game_time(game_time)
                        start_time = max(0, start_time - (clip_duration / 2))  
                        
                        clip_name = f"{match_folder}_{start_time:.1f}.mp4"
                        clip_path = os.path.join(output_root, cls, clip_name)
                        
                        if not os.path.exists(os.path.dirname(clip_path)):
                            os.makedirs(os.path.dirname(clip_path), exist_ok=True)
                            
                        extract_clip(video, start_time, clip_duration, clip_path)
                    except Exception as e:
                        print(f"⚠️ Error processing annotation: {str(e)}")
                        continue
                
                video.close()

def main():
    global classes
    classes = load_classes()
    create_output_dirs(classes)
    
    for split_name in ["val", "test"]:
        process_split(split_name)
    
    print("\n✅ All splits processed successfully!")
    #print(f"Train clips: {output_roots['train']}")
    print(f"Val clips: {output_roots['val']}")
    print(f"Test clips: {output_roots['test']}")

if __name__ == "__main__":
    main()