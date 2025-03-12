import socket
import struct
import numpy as np
import cv2
import threading
import time
from ultralytics import YOLO  # Add import for YOLOv8

class VideoReceiver:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port
        self.socket = None
        self.buffer_size = 65536  # UDP packet buffer size
        self.latest_frame = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize YOLOv8n model
        self.model = YOLO('yolov8n.pt')
        
    def start(self):
        """Start the video receiver server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
        self.socket.bind((self.host, self.port))
        
        self.running = True
        
        # Start receiver thread
        self.receiver_thread = threading.Thread(target=self._receive_frames)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
        
        print(f"Video receiver started on {self.host}:{self.port}")
        
    def stop(self):
        """Stop the video receiver server"""
        self.running = False
        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=1.0)
        
        if self.socket:
            self.socket.close()
        
    def _receive_frames(self):
        """Thread function to receive video frames"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(self.buffer_size)
                
                # First 4 bytes contain the JPEG data length
                data_length = struct.unpack('!I', data[:4])[0]
                
                # The rest is the JPEG data
                jpeg_data = data[4:4+data_length]
                
                # Decode JPEG to numpy array
                img = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if img is not None:
                    with self.frame_lock:
                        self.latest_frame = img
                        self.frame_count += 1
                        
            except Exception as e:
                print(f"Error receiving frame: {e}")
                
    def get_latest_frame(self):
        """Get the latest received frame"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def detect_objects(self, frame):
        """Run YOLOv8n object detection on the frame"""
        if frame is None:
            return None, []
            
        # Run YOLOv8 inference on the frame
        results = self.model(frame)
        
        # Process results
        detected_frame = results[0].plot()  # Get the annotated frame
        detections = []
        
        # Extract detection information
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get box coordinates
                conf = float(box.conf[0])              # Get confidence score
                cls = int(box.cls[0])                  # Get class id
                name = result.names[cls]               # Get class name
                
                detections.append({
                    'class': name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })
        
        return detected_frame, detections
            
    def get_fps(self):
        """Calculate current FPS"""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            return self.frame_count / elapsed_time
        return 0
        
    def reset_fps_counter(self):
        """Reset the FPS counter"""
        self.frame_count = 0
        self.start_time = time.time()


def main():
    # Create and start video receiver
    receiver = VideoReceiver(port=5000)
    receiver.start()
    
    try:
        last_fps_display = time.time()
        fps_display_interval = 1.0  # Update FPS display every second
        
        while True:
            # Get latest frame
            frame = receiver.get_latest_frame()
            
            if frame is not None:
                # Run object detection
                detected_frame, detections = receiver.detect_objects(frame)
                
                # Calculate and display FPS every second
                current_time = time.time()
                if current_time - last_fps_display > fps_display_interval:
                    fps = receiver.get_fps()
                    print(f"Current FPS: {fps:.2f}")
                    
                    # Print detected objects
                    if detections:
                        print(f"Detected objects: {len(detections)}")
                        for i, det in enumerate(detections[:5]):  # Show top 5 detections
                            print(f"  {i+1}. {det['class']} ({det['confidence']:.2f})")
                    
                    receiver.reset_fps_counter()
                    last_fps_display = current_time
                
                # Add FPS text to detected frame
                cv2.putText(detected_frame, f"FPS: {receiver.get_fps():.2f}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the frame with detections
                cv2.imshow('Video Receiver with YOLOv8', detected_frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        receiver.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()