import socket
import struct
import numpy as np
import cv2
from threading import Thread, Event
from ultralytics import YOLO
import time

class UnityVideoReceiver:
    def __init__(self, host='127.0.0.1', port=5555, model_path="yolov8n.pt", reconnect_delay=5):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.stop_event = Event()
        self.latest_frame = None
        self.processing_thread = None
        self.receive_thread = None
        self.reconnect_delay = reconnect_delay

        # Load YOLOv8 model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

    def start_server(self):
        """Start the server to receive video frames from Unity"""
        try:
            while not self.stop_event.is_set():
                try:
                    self._setup_server_socket()
                    self._accept_client_connection()
                    
                    # Start processing thread
                    self.processing_thread = Thread(target=self._process_frames)
                    self.processing_thread.daemon = True
                    self.processing_thread.start()
                    
                    # Receive frames
                    self._receive_frames()
                
                except (ConnectionResetError, BrokenPipeError) as conn_error:
                    print(f"Connection error: {conn_error}")
                except Exception as e:
                    print(f"Unexpected error: {e}")
                
                # Cleanup before potential reconnection
                self.stop_server()
                
                # Wait before attempting to reconnect
                print(f"Waiting {self.reconnect_delay} seconds before reconnecting...")
                time.sleep(self.reconnect_delay)
        
        except KeyboardInterrupt:
            print("Server stopped by user")
        finally:
            self.stop_server()

    def _setup_server_socket(self):
        """Set up the server socket"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.settimeout(10)  # 10-second timeout for accepting connections
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}")

    def _accept_client_connection(self):
        """Accept a client connection"""
        try:
            self.client_socket, addr = self.server_socket.accept()
            self.client_socket.settimeout(10)  # 10-second timeout for receiving data
            print(f"Connection from Unity established: {addr}")
        except socket.timeout:
            print("Connection timeout. Retrying...")
            raise

    def _receive_frames(self):
        """Receive video frames from Unity"""
        while not self.stop_event.is_set():
            try:
                size_data = self._receive_all(4)
                if not size_data:
                    break
                
                frame_size = struct.unpack('<I', size_data)[0]
                frame_data = self._receive_all(frame_size)
                if not frame_data:
                    break
                
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    self.latest_frame = frame
            
            except (ConnectionResetError, BrokenPipeError):
                print("Connection lost. Attempting to reconnect...")
                break
            except Exception as e:
                print(f"Error receiving frames: {e}")
                break

    def _receive_all(self, n):
        """Receive exactly n bytes from the socket"""
        data = bytearray()
        while len(data) < n:
            try:
                packet = self.client_socket.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except socket.timeout:
                print("Receive timeout. Reconnecting...")
                return None
        return data
    
    def _process_frames(self):
        """Process the received frames using YOLOv8"""
        while not self.stop_event.is_set():
            if self.latest_frame is not None:
                try:
                    frame = self.latest_frame.copy()

                    results = self.model(frame)
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0].item()
                            cls = int(box.cls[0].item())
                            label = f"{self.model.names[cls]} {conf:.2f}"

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imshow('YOLOv8 Detection', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop_server()
                        break
                
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    break

    def stop_server(self):
        """Stop the server and release resources"""
        self.stop_event.set()
        
        if self.client_socket:
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
                self.client_socket.close()
            except Exception:
                pass
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    receiver = UnityVideoReceiver()
    receiver.start_server()