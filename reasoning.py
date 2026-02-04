class FocusReasoner:
    def __init__(self, strict=True):
        self.strict = strict

    def analyze(self, temporal_state):
        """
        Converts temporal behavioral signals into
        focus score, state, explanation, and event.
        """

        gaze_outside = temporal_state["gaze_outside_duration"]
        iris_invalid = temporal_state["iris_invalid_duration"]
        face_absent = temporal_state["face_absent_duration"]
        head_turned = temporal_state["head_turned_duration"]
        nose_var = temporal_state["nose_variance"]

        focus_score = self._compute_focus_score(
            gaze_outside,
            iris_invalid,
            face_absent,
            head_turned,
            nose_var
        )

        state = self._classify_state(
            gaze_outside,
            iris_invalid,
            face_absent,
            head_turned,
            focus_score
        )

        explanation = self._generate_explanation(
            gaze_outside,
            iris_invalid,
            face_absent,
            head_turned,
            state
        )

        event = self._detect_event(state)

        return {
            "focus_score": focus_score,
            "state": state,
            "explanation": explanation,
            "event": event
        }

    # ------------------------------------------------------------------

    def _compute_focus_score(
        self,
        gaze_outside,
        iris_invalid,
        face_absent,
        head_turned,
        nose_var
    ):
        score = 100

        # Looking away (eyes visible but gaze diverted)
        if gaze_outside > 1.5:
            score -= min(gaze_outside * 12, 35)

        # Eyes not visible (occlusion, phone, hand)
        if iris_invalid > 1.0:
            score -= min(iris_invalid * 20, 70)

        # Face not visible at all
        if face_absent > 1.0:
            score -= min(face_absent * 25, 80)

        # Head turned away (even if iris visible)
        if head_turned > 1.5:
            score -= min(head_turned * 15, 50)

        # Excessive jitter / abnormal motion
        if nose_var > 15:
            score -= min(nose_var, 20)

        return max(int(score), 0)

    # ------------------------------------------------------------------

    def _classify_state(
        self,
        gaze_outside,
        iris_invalid,
        face_absent,
        head_turned,
        score
    ):
        if face_absent > 2.0:
            return "Face Not Visible"

        if iris_invalid > 2.0:
            return "Eyes Not Visible"

        if head_turned > 3.0:
            return "Head Turned Away"

        if gaze_outside > 3.0:
            return "Distracted"

        if score < 60:
            return "Recovering"

        return "Focused"

    # ------------------------------------------------------------------

    def _detect_event(self, state):
        """
        Determines whether an event should be logged.
        """
        if state in [
            "Face Not Visible",
            "Eyes Not Visible",
            "Head Turned Away"
        ]:
            return "visibility_violation"

        if state == "Distracted":
            return "attention_drop"

        return None

    # ------------------------------------------------------------------

    def _generate_explanation(
        self,
        gaze_outside,
        iris_invalid,
        face_absent,
        head_turned,
        state
    ):
        if state == "Focused":
            return (
                "Attention appears stable with visible eyes "
                "and forward-facing head orientation."
            )

        if state == "Recovering":
            return (
                "Brief attention lapse detected, followed by recovery behavior."
            )

        if state == "Distracted":
            return (
                f"Eyes diverted from screen for "
                f"{round(gaze_outside, 1)} seconds."
            )

        if state == "Eyes Not Visible":
            return (
                f"Eyes not clearly visible for "
                f"{round(iris_invalid, 1)} seconds."
            )

        if state == "Face Not Visible":
            return (
                f"Face not visible for "
                f"{round(face_absent, 1)} seconds."
            )

        if state == "Head Turned Away":
            return (
                f"Head turned away from screen for "
                f"{round(head_turned, 1)} seconds."
            )

        return "Behavioral state updated."
