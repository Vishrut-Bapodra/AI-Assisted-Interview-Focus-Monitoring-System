import cv2, time
from vision import VisionProcessor
from temporal import TemporalAnalyzer
from reasoning import FocusReasoner
from logger import EventLogger


def draw_overlay(frame, decision):

    h, w = frame.shape[:2]

    focus_score = decision["focus_score"]
    state = decision["state"]
    explanation = decision["explanation"]

    color = (0,255,0)
    if state in ["Distracted", "Recovering"]:
        color = (0, 165, 255)
    elif state == "Sustained Distraction":
        color = (0,0,255)

    cv2.putText(
        frame,
        f"Focus Score: {focus_score}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        frame,
        f"State: {state}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        frame,
        explanation,
        (20, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FPS, 10)
    vision = VisionProcessor()
    temporal = TemporalAnalyzer(fps=10)
    reasoner = FocusReasoner()
    logger= EventLogger()

    print("[INFO] Interview Focus Monitor started.")
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read frame from webcam")
            break

        signals = vision.extract_signals(frame)
        temporal_state = temporal.update(signals)
        decision = reasoner.analyze(temporal_state)

        draw_overlay(frame, decision)

        if decision["event"]:
            logger.log_event(
                frame=frame,
                event_type=decision["event"],
                focus_score=decision["focus_score"],
                metadata={
                    "state": decision["state"],
                    "explanation": decision["explanation"]
                }
            )

        cv2.imshow("AI Interview Focus Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

            time.sleep(0.01)
        
    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] Session ended.")
    print("[INFO] Logs saved to data/ directory.")

if __name__ == "__main__":
    main()