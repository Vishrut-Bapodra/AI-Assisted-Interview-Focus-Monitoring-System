import os, json, cv2
from datetime import datetime

class EventLogger:
    def __init__(self, session_id=None, base_dir="data"):
        
        self.session_id = session_id or self._generate_session_id()
        self.base_dir = base_dir
        self.session_dir = os.path.join(self.base_dir, self.session_id)
        self.screenshots_dir = os.path.join(self.session_dir, "screenshots")
        self.log_file_path = os.path.join(self.session_dir, "events.json")
        self._create_directories()
        self._init_log_file()

    def _generate_session_id(self):
        return datetime.now().strftime("session_%Y%m%d_%H%M%S")
    
    def _create_directories(self):
        os.makedirs(self.screenshots_dir, exist_ok=True)

    def _init_log_file(self):
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, "w") as f:
                json.dump([], f, indent=4)
    
    def log_event(self, frame, event_type, focus_score, metadata=None):

        timestamp = datetime.now().strftime("%H:%M:%S")
        screenshot_path = self._save_screenshot(frame, event_type, timestamp)

        event_data = {
            "event_type": event_type,
            "timestamp": timestamp,
            "focus_score": focus_score,
            "screenshot": screenshot_path,
            "metadata": metadata or {}
        }

        self._write_event(event_data)

    def _save_screenshot(self, frame, event_type, timestamp):
        filename = f"{event_type}_{timestamp.replace(':', "-")}.jpg"
        path = os.path.join(self.screenshots_dir, filename)
        cv2.imwrite(path, frame)
        return path
    
    def _write_event(self, event_data):
        with open(self.log_file_path, "r+") as f:
            data = json.load(f)
            data.append(event_data)
            f.seek(0)
            json.dump(data, f, indent=4)
        
    def get_session_summary(self):
        with open(self.log_file_path, "r") as f:
            return json.load(f)