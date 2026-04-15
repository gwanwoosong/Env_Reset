"""
Reset Manager: orchestrates the full reset pipeline (Phase 1~5).

Usage:
    from table_reset.reset_manager import ResetManager

    manager = ResetManager(env)
    success = manager.reset()
"""

import numpy as np

from table_reset.config import (
    TOP_POSE, E_T_C, GRID_SLOTS,
    MAX_VERIFICATION_LOOPS, MAX_GRASP_RETRIES,
    ALL_OBJECTS,
)
from table_reset.motion import move_to_top, pick_and_place, run
from table_reset.perception import perceive_scene, SceneState
from table_reset.grasp_planner import (
    create_anygrasp, plan_grasp_outside, plan_grasp_zone,
)


class ResetManager:
    """Manages the full table reset pipeline.

    Attributes:
        env: DROID RobotEnv instance
        processor: Sam3Processor instance
        anygrasp: loaded AnyGrasp model
        used_slots: set of grid slot indices currently occupied
    """

    def __init__(self, env, processor=None, anygrasp=None):
        """Initialize the ResetManager.

        Args:
            env: DROID RobotEnv (already initialized)
            processor: Sam3Processor (if None, will be created on first use)
            anygrasp: AnyGrasp instance (if None, will be created on first use)
        """
        self.env = env
        self.processor = processor
        self.anygrasp = anygrasp
        self.used_slots = set()

    def _ensure_models(self):
        """Lazy-load models if not provided at init."""
        if self.processor is None:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            model = build_sam3_image_model()
            self.processor = Sam3Processor(model)
            print("[ResetManager] SAM3 loaded.")

        if self.anygrasp is None:
            self.anygrasp = create_anygrasp()

    def _next_grid_slot(self):
        """Get the next available grid slot.

        Returns:
            [x, y, z] of the next free slot, or None if all slots are full.
        """
        for i, slot in enumerate(GRID_SLOTS):
            if i not in self.used_slots:
                self.used_slots.add(i)
                return slot
        print("[ResetManager] All grid slots occupied!")
        return None

    def _reset_grid(self):
        """Reset grid slot tracking for a new reset cycle."""
        self.used_slots = set()

    def reset(self):
        """Execute the full reset pipeline.

        Phases:
            1. Perception (SAM3 detection)
            2. Zone classification
            3. Build move queue
            4. Sequential pick-and-place
            5. Verification (loop back to Phase 1 if needed)

        Returns:
            True if all objects are in the Allowed Zone, False otherwise.
        """
        self._ensure_models()
        self._reset_grid()

        for iteration in range(MAX_VERIFICATION_LOOPS):
            print(f"\n{'='*60}")
            print(f"[ResetManager] Iteration {iteration + 1}/{MAX_VERIFICATION_LOOPS}")
            print(f"{'='*60}")

            # Phase 1-3: Perceive and plan
            move_to_top(self.env)
            scene = perceive_scene(self.env, self.processor, E_T_C)

            if not scene.move_queue:
                # Check if we have enough objects in allowed zone
                n_allowed = len(scene.objects_in_allowed)
                print(f"[ResetManager] No objects to move. "
                      f"{n_allowed}/{len(ALL_OBJECTS)} in Allowed Zone.")
                if n_allowed >= len(ALL_OBJECTS):
                    print("[ResetManager] Reset complete!")
                    return True
                elif iteration < MAX_VERIFICATION_LOOPS - 1:
                    print("[ResetManager] Some objects not detected. Re-scanning...")
                    continue
                else:
                    print("[ResetManager] Max iterations reached with missing objects.")
                    return False

            # Phase 4: Sequential pick-and-place
            for det in scene.move_queue:
                print(f"\n--- Processing: {det.name} (zone: {det.zone}) ---")

                target_slot = self._next_grid_slot()
                if target_slot is None:
                    print("[ResetManager] No grid slots available. Stopping.")
                    break

                success = self._pick_and_place_object(det, target_slot)
                if not success:
                    print(f"[ResetManager] Failed to move {det.name}. "
                          f"Will retry in next iteration.")
                    # Free the slot since we didn't use it
                    self.used_slots.discard(max(self.used_slots))

            # Phase 5: Verification — loop back to Phase 1
            print(f"\n[ResetManager] Iteration {iteration + 1} complete. "
                  f"Verifying...")

        # Final check
        move_to_top(self.env)
        scene = perceive_scene(self.env, self.processor, E_T_C)
        n_allowed = len(scene.objects_in_allowed)
        n_total = len(ALL_OBJECTS)

        if not scene.move_queue:
            print(f"[ResetManager] Final verification: {n_allowed}/{n_total} "
                  f"objects in Allowed Zone. Reset complete!")
            return True
        else:
            print(f"[ResetManager] {len(scene.move_queue)} objects still "
                  f"outside Allowed Zone after {MAX_VERIFICATION_LOOPS} iterations.")
            print("[ResetManager] Human intervention required.")
            return False

    def _pick_and_place_object(self, detection, target_slot):
        """Pick a single object and place it in the target grid slot.

        Uses zone-based grasp planning:
          - OUTSIDE: individual bbox crop
          - YELLOW_PLATE / BLUE_TRAY: zone-wide crop with surface filtering

        Args:
            detection: Detection object to move
            target_slot: [x, y, z] target position

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(MAX_GRASP_RETRIES):
            print(f"  Grasp attempt {attempt + 1}/{MAX_GRASP_RETRIES} "
                  f"for {detection.name}")

            # Move to observation pose before each attempt
            move_to_top(self.env)

            # Plan grasp based on zone
            if detection.zone in ("YELLOW_PLATE", "BLUE_TRAY"):
                grasp_pose = plan_grasp_zone(
                    self.env, self.anygrasp, detection.zone)
            else:
                grasp_pose = plan_grasp_outside(
                    self.env, self.anygrasp, detection)

            if grasp_pose is None:
                print(f"  No grasp found for {detection.name}.")
                continue

            print(f"  Grasp pose: [{grasp_pose[0]:.3f}, {grasp_pose[1]:.3f}, "
                  f"{grasp_pose[2]:.3f}]")

            # Execute pick and place
            success = pick_and_place(self.env, grasp_pose, target_slot)
            if success:
                print(f"  Successfully moved {detection.name} to slot.")
                return True
            else:
                print(f"  Pick-and-place failed for {detection.name}.")

        print(f"  All {MAX_GRASP_RETRIES} attempts failed for {detection.name}.")
        return False
