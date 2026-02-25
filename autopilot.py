import copy
import numpy as np
from collections import deque
from datetime import datetime

class TrainingAutopilot:
    def __init__(
        self,
        optimizer,
        min_steps=4000,
        window=300,
        eval_steps=500,
        improvement_threshold=0.02,
        log_file="autopilot.log"
    ):
        self.optimizer = optimizer
        self.min_steps = min_steps
        self.window = window
        self.eval_steps = eval_steps
        self.improvement_threshold = improvement_threshold

        self.val_hist = deque(maxlen=window)
        self.grad_hist = deque(maxlen=window)

        self.state = "OBSERVE"
        self.last_action_step = None
        self.saved_state = None
        self.baseline_loss = None
        self.active_param = None

        self.log_file = log_file

    def log(self, msg, icon="🔵"):
        t = datetime.now().strftime("%H:%M:%S")
        line = f"[{t}] {icon} {msg}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def record(self, val_loss, grad_norm):
        self.val_hist.append(val_loss)
        self.grad_hist.append(grad_norm)

    def ready(self, step):
        if step < self.min_steps:
            return False
        if len(self.val_hist) < self.window:
            return False
        return True

    def val_slope(self):
        y = np.array(self.val_hist)
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]

    def grad_stable(self):
        g = np.array(self.grad_hist)
        return np.std(g) < 0.3

    def maybe_arm(self, step):
        if self.state != "OBSERVE":
            return

        if self.ready(step) and self.grad_stable():
            slope = self.val_slope()
            if abs(slope) < 1e-4:
                self.state = "ARMED"
                self.log("Autopilot ARMED (late-stage detected)", "🟡")

    def act_lr_decay(self, step):
        self.saved_state = copy.deepcopy(self.optimizer.state_dict())
        self.baseline_loss = np.mean(self.val_hist)
        self.active_param = "lr"

        for g in self.optimizer.param_groups:
            g["lr"] *= 0.7

        self.last_action_step = step
        self.state = "EVAL"
        self.log("Applied LR decay ×0.7", "🟢")

    def evaluate(self, step):
        if step - self.last_action_step < self.eval_steps:
            return

        new_loss = np.mean(self.val_hist)
        delta = self.baseline_loss - new_loss

        if delta >= self.improvement_threshold:
            self.log(f"Change kept (Δval={delta:.4f})", "🟢")
            self.state = "OBSERVE"
            self.saved_state = None
        else:
            self.optimizer.load_state_dict(self.saved_state)
            self.log("Change reverted (no improvement)", "🔴")
            self.state = "OBSERVE"
            self.saved_state = None
