import signal
import sys
import torch
import threading
from pathlib import Path
from typing import Optional, Callable


class TrainingHandler:
    """Handler for managing training interruptions and checkpoint saving."""

    def __init__(self, checkpoint_path: Path):
        """Initialize the training handler.

        Args:
            checkpoint_path: Path where checkpoints will be saved
        """
        self.checkpoint_path = checkpoint_path
        self._interrupt_requested = False
        self._pause_lock = threading.Lock()
        self._save_in_progress = False
        self._original_sigint = None
        self._setup_signal_handler()

    def _setup_signal_handler(self):
        """Set up the SIGINT (Ctrl+C) signal handler."""
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signal gracefully.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        with self._pause_lock:
            if self._interrupt_requested:
                # Second interrupt - force exit
                print("\n\n⚠️  Force exit requested. Exiting immediately...")
                if self._original_sigint:
                    self._original_sigint(signum, frame)
                else:
                    sys.exit(1)

            self._interrupt_requested = True
            print("\n\n⏸️  Interrupt received. Pausing training and saving checkpoint...")
            print("   Press Ctrl+C again to force exit.")

    def is_interrupted(self) -> bool:
        """Check if an interrupt has been requested.

        Returns:
            True if interrupt was requested, False otherwise
        """
        return self._interrupt_requested

    def reset_interrupt(self):
        """Reset the interrupt flag for continued training."""
        with self._pause_lock:
            self._interrupt_requested = False

    def save_checkpoint(self, model, optimizer, epoch: int, step: int,
                       vocab, block_size: int, loss: Optional[float] = None):
        """Save a training checkpoint.

        Args:
            model: The model state dict
            optimizer: The optimizer state dict
            epoch: Current epoch number
            step: Current step number
            vocab: Vocabulary object
            block_size: Block size used for training
            loss: Current loss value (optional)
        """
        with self._pause_lock:
            self._save_in_progress = True

        try:
            checkpoint = {
                "model": model,
                "optimizer": optimizer,
                "epoch": epoch,
                "step": step,
                "chars": list(vocab.stoi.keys()),
                "block_size": block_size,
            }
            if loss is not None:
                checkpoint["last_loss"] = loss

            # Save to the configured checkpoint path
            torch.save(checkpoint, self.checkpoint_path)

            # Also save a backup copy with timestamp info
            backup_path = self.checkpoint_path.parent / f"{self.checkpoint_path.stem}_backup_epoch{epoch}_step{step}{self.checkpoint_path.suffix}"
            torch.save(checkpoint, backup_path)

            print(f"\n✓ Checkpoint saved to: {self.checkpoint_path}")
            print(f"✓ Backup saved to: {backup_path}")
            print(f"   Epoch: {epoch}, Step: {step}" + (f", Loss: {loss:.4f}" if loss is not None else ""))

        except Exception as e:
            print(f"\n✗ Error saving checkpoint: {e}", file=sys.stderr)
        finally:
            with self._pause_lock:
                self._save_in_progress = False

    def wait_for_save_completion(self, timeout: float = 30.0):
        """Wait for any ongoing save operation to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if save completed, False if timeout occurred
        """
        import time
        start_time = time.time()
        while self._save_in_progress:
            if time.time() - start_time > timeout:
                print("\n⚠️  Timeout waiting for save operation to complete.", file=sys.stderr)
                return False
            time.sleep(0.1)
        return True

    def restore_signal_handler(self):
        """Restore the original signal handler."""
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore signal handler."""
        self.restore_signal_handler()
        # Wait for any pending saves
        self.wait_for_save_completion(timeout=5.0)


def load_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to

    Returns:
        Dictionary containing checkpoint data, or None if loading failed
    """
    if not checkpoint_path.exists():
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✓ Loaded checkpoint from: {checkpoint_path}")
        return checkpoint
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}", file=sys.stderr)
        return None


def can_resume_training(checkpoint_path: Path) -> bool:
    """Check if training can be resumed from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        True if checkpoint exists and is valid, False otherwise
    """
    if not checkpoint_path.exists():
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        required_keys = ["model", "epoch", "chars", "block_size"]
        return all(key in checkpoint for key in required_keys)
    except Exception:
        return False