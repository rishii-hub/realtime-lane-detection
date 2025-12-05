import cv2
import numpy as np
from collections import deque
import time

class RealtimeLaneDetector:
    def __init__(self, source=0, resolution=(640, 480)):
        """
        Initialize real-time lane detector
        source: 0 for webcam, or video file path, or phone IP for IP webcam
        resolution: (width, height) for camera capture
        """
        self.cap = cv2.VideoCapture(source)
        
        # Set camera resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Temporal smoothing (smaller buffer for faster response)
        self.left_lane_history = deque(maxlen=5)
        self.right_lane_history = deque(maxlen=5)
        
        # Performance tracking
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.processing_time = 0
        
        # Processing modes
        self.show_debug = False
        self.show_edges = False
        self.paused = False
        
    def preprocess_frame(self, frame):
        """Fast preprocessing for real-time performance"""
        # Resize if frame is too large (optional optimization)
        if frame.shape[1] > 800:
            scale = 800 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, 
                             interpolation=cv2.INTER_LINEAR)
        return frame
    
    def canny_edge_fast(self, image):
        """Optimized edge detection for real-time processing"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple Gaussian blur (faster than bilateral)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply CLAHE for better contrast in varying lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)
        
        # Canny edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        return edges
    
    def region_of_interest(self, image):
        """Dynamic ROI based on frame dimensions"""
        height, width = image.shape[:2]
        
        # Trapezoid mask - adjust these percentages based on camera angle
        polygons = np.array([[
            (int(width * 0.05), height),           # Bottom left
            (int(width * 0.95), height),           # Bottom right
            (int(width * 0.60), int(height * 0.6)), # Top right
            (int(width * 0.40), int(height * 0.6))  # Top left
        ]], dtype=np.int32)
        
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        
        # Show ROI boundary in debug mode
        if self.show_debug:
            cv2.polylines(image, polygons, True, (255, 0, 0), 2)
        
        masked = cv2.bitwise_and(image, mask)
        return masked
    
    def detect_lines_fast(self, edges):
        """Optimized Hough line detection"""
        lines = cv2.HoughLinesP(
            edges,
            rho=2,              # Slightly coarser for speed
            theta=np.pi/180,
            threshold=40,        # Lower threshold for real-time
            minLineLength=30,    # Shorter minimum
            maxLineGap=100
        )
        return lines
    
    def average_slope_intercept(self, image, lines):
        """Fast lane averaging with outlier rejection"""
        if lines is None:
            return None
        
        left_lines = []
        right_lines = []
        height = image.shape[0]
        width = image.shape[1]
        
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            
            if x2 == x1:  # Vertical line
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            # Filter unrealistic slopes
            if abs(slope) < 0.4 or abs(slope) > 2.5:
                continue
            
            # Classify by slope and position
            if slope < 0 and x1 < width * 0.55:  # Left lane
                left_lines.append((slope, intercept))
            elif slope > 0 and x1 > width * 0.45:  # Right lane
                right_lines.append((slope, intercept))
        
        # Average lanes
        lane_lines = []
        
        for lines_list in [left_lines, right_lines]:
            if lines_list:
                slope, intercept = np.median(lines_list, axis=0)
                y1 = height
                y2 = int(height * 0.6)
                x1 = int((y1 - intercept) / slope)
                x2 = int((y2 - intercept) / slope)
                
                # Validate coordinates
                if -width < x1 < width * 2 and -width < x2 < width * 2:
                    lane_lines.append([x1, y1, x2, y2])
        
        return lane_lines if lane_lines else None
    
    def smooth_lanes(self, current_lanes):
        """Quick temporal smoothing"""
        if current_lanes is None or len(current_lanes) == 0:
            # Use recent history
            smoothed = []
            if len(self.left_lane_history) > 0:
                smoothed.append(np.mean(self.left_lane_history, axis=0).astype(int).tolist())
            if len(self.right_lane_history) > 0:
                smoothed.append(np.mean(self.right_lane_history, axis=0).astype(int).tolist())
            return smoothed if smoothed else None
        
        # Update history
        if len(current_lanes) >= 1:
            self.left_lane_history.append(current_lanes[0])
        if len(current_lanes) >= 2:
            self.right_lane_history.append(current_lanes[1])
        
        return current_lanes
    
    def visualize(self, image, lanes, edges=None):
        """Fast visualization"""
        output = image.copy()
        
        if lanes and len(lanes) >= 2:
            # Fill lane area
            pts = np.array([[
                [lanes[0][0], lanes[0][1]],
                [lanes[0][2], lanes[0][3]],
                [lanes[1][2], lanes[1][3]],
                [lanes[1][0], lanes[1][1]]
            ]], dtype=np.int32)
            
            overlay = output.copy()
            cv2.fillPoly(overlay, pts, (0, 255, 0))
            output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
            
            # Draw lane lines
            for lane in lanes:
                x1, y1, x2, y2 = lane
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 10)
            
            # Calculate deviation
            lane_center = (lanes[0][0] + lanes[1][0]) // 2
            vehicle_center = image.shape[1] // 2
            deviation = vehicle_center - lane_center
            
            # Deviation indicator
            color = (0, 255, 0) if abs(deviation) < 50 else (0, 165, 255)
            direction = "RIGHT" if deviation > 0 else "LEFT"
            cv2.putText(output, f"Offset: {abs(deviation)}px {direction}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        elif lanes and len(lanes) == 1:
            # Draw single detected lane
            x1, y1, x2, y2 = lanes[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 10)
            cv2.putText(output, "Single Lane Detected", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(output, "No Lanes Detected", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show edges if requested
        if self.show_edges and edges is not None:
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            output = np.hstack((output, edges_colored))
        
        return output
    
    def add_overlay(self, image):
        """Add performance metrics overlay"""
        # Calculate average FPS
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (280, 120), (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        # Text info
        cv2.putText(image, f"FPS: {avg_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Processing: {self.processing_time:.1f}ms", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Controls hint
        cv2.putText(image, "Q:Quit D:Debug E:Edges P:Pause", 
                   (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        return image
    
    def run(self):
        """Main real-time processing loop"""
        print("Starting Real-Time Lane Detection...")
        print("Controls:")
        print("  Q - Quit")
        print("  D - Toggle debug view")
        print("  E - Toggle edge view")
        print("  P - Pause/Resume")
        print("  S - Save screenshot")
        
        if not self.cap.isOpened():
            print("Error: Could not open camera/video source")
            return
        
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of stream or error reading frame")
                    break
                
                start_time = time.time()
                
                # Preprocess
                frame = self.preprocess_frame(frame)
                
                # Edge detection
                edges = self.canny_edge_fast(frame)
                
                # Apply ROI
                cropped = self.region_of_interest(edges)
                
                # Detect lines
                lines = self.detect_lines_fast(cropped)
                
                # Average lanes
                averaged_lines = self.average_slope_intercept(frame, lines)
                
                # Smooth lanes
                smoothed_lanes = self.smooth_lanes(averaged_lines)
                
                # Visualize
                output = self.visualize(frame, smoothed_lanes, edges)
                
                # Add overlay
                output = self.add_overlay(output)
                
                # Calculate processing time
                end_time = time.time()
                self.processing_time = (end_time - start_time) * 1000
                fps = 1 / (end_time - start_time) if end_time > start_time else 0
                self.fps_history.append(fps)
                
                self.frame_count += 1
                current_frame = output
            
            # Display
            cv2.imshow("Real-Time Lane Detection", current_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                print(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")
            elif key == ord('e'):
                self.show_edges = not self.show_edges
                print(f"Edge view: {'ON' if self.show_edges else 'OFF'}")
            elif key == ord('p'):
                self.paused = not self.paused
                print(f"{'PAUSED' if self.paused else 'RESUMED'}")
            elif key == ord('s'):
                filename = f'lane_capture_{int(time.time())}.jpg'
                cv2.imwrite(filename, current_frame)
                print(f"Saved: {filename}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession stats:")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Average FPS: {np.mean(self.fps_history):.2f}")

# ========================================
# USAGE EXAMPLES
# ========================================

def main():
    """Main function with different source options"""
    
    print("Choose input source:")
    print("1. Webcam (default)")
    print("2. Video file")
    print("3. IP Webcam (phone)")
    
    choice = input("Enter choice (1-3) or press Enter for webcam: ").strip()
    
    if choice == '2':
        video_path = input("Enter video file path: ")
        detector = RealtimeLaneDetector(source=video_path, resolution=(640, 480))
    elif choice == '3':
        ip = input("Enter IP webcam URL (e.g., http://192.168.1.100:8080/video): ")
        detector = RealtimeLaneDetector(source=ip, resolution=(640, 480))
    else:
        # Webcam (default)
        detector = RealtimeLaneDetector(source=0, resolution=(640, 480))
    
    detector.run()

# For direct webcam usage
def webcam_quick_start():
    """Quick start with webcam"""
    detector = RealtimeLaneDetector(source=0, resolution=(640, 480))
    detector.run()

# For video file usage
def video_file(path):
    """Process video file"""
    detector = RealtimeLaneDetector(source=path, resolution=(640, 480))
    detector.run()

# For IP webcam (phone camera)
def ip_webcam(url):
    """
    Use phone as camera via IP Webcam app
    Example: http://192.168.1.100:8080/video
    """
    detector = RealtimeLaneDetector(source=url, resolution=(640, 480))
    detector.run()

if __name__ == "__main__":
    main()
    
    # Or use these shortcuts:
    # webcam_quick_start()
    # video_file("test2.mp4")
    # ip_webcam("http://192.168.1.100:8080/video")