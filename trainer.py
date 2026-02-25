import os, math, glob, torch
from torch.utils.data import DataLoader
from autopilot import TrainingAutopilot

# ======================
# EMA
# ======================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)


# ======================
# TRAINER (FP32)
# ======================
class TrainerFP32:
    def __init__(
        self,
        model,
        train_ds,
        val_ds,
        batch_size,
        base_lr,
        warmup_steps,
        max_steps,
        grad_clip,
        device,
        ckpt_dir="checkpoints",
    ):
        self.device = device
        self.model = model.to(device)

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, drop_last=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=0.01
        )

        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip

        self.step = 0
        self.best_val = float("inf")

        self.ema = EMA(self.model)
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------
    def lr_schedule(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    

    # ------------------
    def train(self):
        self.model.train()
        self.val_interval = 1000

        autopilot = TrainingAutopilot(self.optimizer)

        for x, y in self.train_loader:
            self.step += 1

            # LR update
            """order of operations:
            forward
            backward
            clip
            optimizer.step()
            EMA.update()
    
            """
            lr = self.base_lr * self.lr_schedule(self.step)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            x, y = x.to(self.device), y.to(self.device)

            # -------- forward --------
            _, loss = self.model(x, y, pad_token_id=0)

            if not torch.isfinite(loss):
                print(f"❌ NaN/Inf loss at step {self.step}, stopping")
                return

            # -------- backward --------
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )

            self.optimizer.step()
            self.ema.update(self.model)

            # -------- logging --------
            if self.step % 100 == 0:
                ppl = math.exp(loss.item())
                print(
                    f"step {self.step} | loss {loss.item():.4f} | "
                    f"ppl {ppl:.2f} | grad_norm {grad_norm:.3f}"
                )

            if self.step % self.val_interval == 0:
                val_loss = self.validate()
                self.save(self.step // self.val_interval, val_loss)

                autopilot.record(val_loss, grad_norm)
                autopilot.maybe_arm(self.step)

                if autopilot.state == "ARMED":
                    autopilot.act_lr_decay(self.step)

                if autopilot.state == "EVAL":
                    autopilot.evaluate(self.step)

            if self.step >= self.max_steps:
                return
            
    def load_checkpoint(self, path, load_optimizer=False):
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model"])
        self.step = ckpt.get("step", 0)

        if load_optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scaler.load_state_dict(ckpt["scaler"])

        print(f"✅ Loaded checkpoint: {path} (step={self.step})")
            
    def save(self, epoch, val_loss):
        if val_loss < self.best_val:
            self.best_val = val_loss

            path = os.path.join(self.ckpt_dir, "best.pt")

            torch.save(
                {
                    "model": self.model.state_dict(),
                    "ema": self.ema.shadow,
                    "step": self.step,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                path,
            )

            print(f"🏆 Saved BEST checkpoint (val={val_loss:.4f}) → {path}")

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        losses = []

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y, pad_token_id=0)
            losses.append(loss.item())

        avg = sum(losses) / len(losses)
        ppl = math.exp(avg)

        print(f"📊 val loss {avg:.4f} | ppl {ppl:.2f}")

        self.model.train()
        return avg
    
    def auto_resume(self):
        """Automatically find and load the best checkpoint if it exists"""
        
        # Look for best checkpoint first
        best_path = f"{self.ckpt_dir}/best.pt"
        if os.path.exists(best_path):
            print(f"🔄 Found best checkpoint: {best_path}")
            try:
                self.load_checkpoint(best_path)
                return True
            except RuntimeError as e:
                if "architecture mismatch" in str(e):
                    print(f"⚠️  {e}")
                    # Rename old checkpoint to preserve it
                    backup_path = f"{self.ckpt_dir}/best_old_architecture.pt"
                    os.rename(best_path, backup_path)
                    print(f"📦 Backed up incompatible checkpoint to {backup_path}")
                    print("🆕 Starting fresh training with new architecture")
                    return False
                raise
            
        # Look for the latest checkpoint with validation loss in filename
        pattern = f"{self.ckpt_dir}/transformer_run_v1_best_val_*.pt"
        ckpts = glob.glob(pattern)
        if ckpts:
            # Sort by modification time to get the most recent
            latest_ckpt = max(ckpts, key=os.path.getmtime)
            print(f"🔄 Found checkpoint: {latest_ckpt}")
            try:
                self.load_checkpoint(latest_ckpt)
                return True
            except RuntimeError as e:
                if "architecture mismatch" in str(e):
                    print(f"⚠️  {e}")
                    print("🆕 Starting fresh training with new architecture")
                    return False
                raise
            
        # Look for any .pt files in checkpoints
        pattern = f"{self.ckpt_dir}/*.pt"
        ckpts = glob.glob(pattern)
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getmtime)
            print(f"🔄 Found checkpoint: {latest_ckpt}")
            try:
                self.load_checkpoint(latest_ckpt)
                return True
            except RuntimeError as e:
                if "architecture mismatch" in str(e):
                    print(f"⚠️  {e}")
                    print("🆕 Starting fresh training with new architecture")
                    return False
                raise
            
        print("🆕 No checkpoint found, starting fresh training")
        return False

