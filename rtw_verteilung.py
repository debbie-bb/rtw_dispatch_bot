import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EmergencyDispatchEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode=None,
        max_steps=200,
        p_call=0.3,  # Wahrscheinlichkeit für neuen Call pro Schritt
        max_busy=5,
        service_time_base=2,
        waiting_penalty=0.5,
        success_reward_base=5.0,
        move_towards_bonus=0.3,
        move_away_penalty=0.1,
        invalid_dispatch_penalty=0.1,
        busy_move_penalty=0.05,
        timestep_penalty=0.01,
        lost_call_penalty=10.0,
    ):

        super().__init__()
        
        # Parameter
        self.num_zones = 3   # Zonen: 0,1,2
        self.max_busy = max_busy
        self.max_steps = max_steps
        self.p_call = p_call
        self.service_time_base = service_time_base
        self.waiting_penalty = waiting_penalty               # Warte Penalty pro Zeitschritt für offenen Call
        self.success_reward_base = success_reward_base       # Grundreward pro Prio beim erfolgreichen Dispatch
        self.move_towards_bonus = move_towards_bonus         # Bonus, wenn RTW sich dem aktiven Call nähert
        self.move_away_penalty = move_away_penalty           # kleine Strafe, wenn RTW sich weiter entfernt
        self.invalid_dispatch_penalty = invalid_dispatch_penalty  # Strafe für Dispatch ohne gültigen Call / busy
        self.busy_move_penalty = busy_move_penalty           # Strafe bei Bewegung während busy
        self.timestep_penalty = timestep_penalty             # pro Zeitschritt
        self.lost_call_penalty = lost_call_penalty           # Faktor für verlorene Calls pro Prio

        # Observation space: [amb0_pos, amb1_pos, amb0_busy , amb1_busy, call_zone, call_priority]
        self.observation_space = spaces.MultiDiscrete([
            self.num_zones,           # amb0_pos
            self.max_busy + 1,        # amb0_busy (0..max_busy)
            self.num_zones,           # amb1_pos
            self.max_busy + 1,        # amb1_busy
            self.num_zones + 1,       # call_zone (0..11, 12 = none)
            4                         # call_priority (0,1,2,3) 3 = none
        ])
        
        # Actions: 0=nichts, 1=amb0 dispatch, 2=amb1 dispatch, 3=amb0 left,
        #          4=amb0 right, 5=amb1 left, 6=amb1 right
        self.action_space = spaces.Discrete(7)
        
        self.render_mode = render_mode
        self.rng = np.random.default_rng()
        
        self.reset()

    def _sample_call(self):
        """Stochastischen Call generieren."""
        if self.rng.random() < self.p_call:
            zone = self.rng.integers(0, self.num_zones)
            prio = self.rng.integers(0, 3)  # 0,1,2
            return zone, prio
        else:
            return None, None

    def _get_obs(self):
        call_zone = self.call_zone if self.call_active else self.num_zones
        call_prio = self.call_priority if self.call_active else 3
        return np.array([
            self.amb0_pos,
            self.amb0_busy,
            self.amb1_pos,
            self.amb1_busy,
            call_zone,
            call_prio
        ], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        
        # Startpositionen
        self.amb0_pos = 0
        self.amb1_pos = self.num_zones - 1
        self.amb0_busy = 0
        self.amb1_busy = 0
        
        self.call_active = False
        self.call_zone = self.num_zones
        self.call_priority = 3
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1
        reward = 0.0

    # Distanz vor der Aktion für Bewegungs-Shaping merken
        if self.call_active:
            old_dist0 = abs(self.amb0_pos - self.call_zone)
            old_dist1 = abs(self.amb1_pos - self.call_zone)
            prio_weight = float(self.call_priority + 1)
        else:
            old_dist0 = old_dist1 = None
            prio_weight = 0.0

        # 1) Busy-Zeiten updaten
        if self.amb0_busy > 0:
            self.amb0_busy -= 1
        if self.amb1_busy > 0:
            self.amb1_busy -= 1

        # 2) Aktion ausführen
        if action == 0:
            # Nichts tun
            pass

        elif action == 1:
            # amb0 dispatch
            if self.call_active and self.amb0_busy == 0:
                distance = abs(self.amb0_pos - self.call_zone)

                # Positiver Reward für erfolgreich bedienten Call
                reward += self.success_reward_base * prio_weight
                # Kosten für Fahrtstrecke
                reward -= prio_weight * distance

                # Ambulance wird busy
                self.amb0_busy = min(self.max_busy, distance + self.service_time_base)
                # Ambulance endet in Call-Zone
                self.amb0_pos = self.call_zone
                # Call erledigt
                self.call_active = False
                self.call_zone = self.num_zones
                self.call_priority = 3
            else:
                # ungültig
                reward -= self.invalid_dispatch_penalty

        elif action == 2:
            # amb1 dispatch
            if self.call_active and self.amb1_busy == 0:
                distance = abs(self.amb1_pos - self.call_zone)

                reward += self.success_reward_base * prio_weight
                reward -= prio_weight * distance

                self.amb1_busy = min(self.max_busy, distance + self.service_time_base)
                self.amb1_pos = self.call_zone
                self.call_active = False
                self.call_zone = self.num_zones
                self.call_priority = 3
            else:
                reward -= self.invalid_dispatch_penalty

        elif action == 3:
            # amb0 left
            if self.amb0_busy == 0:
                old_dist = old_dist0
                self.amb0_pos = max(0, self.amb0_pos - 1)
                if self.call_active and old_dist is not None:
                    new_dist = abs(self.amb0_pos - self.call_zone)
                    if new_dist < old_dist:
                        reward += self.move_towards_bonus * prio_weight
                    elif new_dist > old_dist:
                        reward -= self.move_away_penalty * prio_weight
            else:
                reward -= self.busy_move_penalty

        elif action == 4:
            # amb0 right
            if self.amb0_busy == 0:
                old_dist = old_dist0
                self.amb0_pos = min(self.num_zones - 1, self.amb0_pos + 1)
                if self.call_active and old_dist is not None:
                    new_dist = abs(self.amb0_pos - self.call_zone)
                    if new_dist < old_dist:
                        reward += self.move_towards_bonus * prio_weight
                    elif new_dist > old_dist:
                        reward -= self.move_away_penalty * prio_weight
            else:
                reward -= self.busy_move_penalty

        elif action == 5:
            # amb1 left
            if self.amb1_busy == 0:
                old_dist = old_dist1
                self.amb1_pos = max(0, self.amb1_pos - 1)
                if self.call_active and old_dist is not None:
                    new_dist = abs(self.amb1_pos - self.call_zone)
                    if new_dist < old_dist:
                        reward += self.move_towards_bonus * prio_weight
                    elif new_dist > old_dist:
                        reward -= self.move_away_penalty * prio_weight
            else:
                reward -= self.busy_move_penalty

        elif action == 6:
            # amb1 right
            if self.amb1_busy == 0:
                old_dist = old_dist1
                self.amb1_pos = min(self.num_zones - 1, self.amb1_pos + 1)
                if self.call_active and old_dist is not None:
                    new_dist = abs(self.amb1_pos - self.call_zone)
                    if new_dist < old_dist:
                        reward += self.move_towards_bonus * prio_weight
                    elif new_dist > old_dist:
                        reward -= self.move_away_penalty * prio_weight
            else:
                reward -= self.busy_move_penalty

        # 3) Waiting Penalty für offenen Call
        if self.call_active:
            prio_weight_wait = float(self.call_priority + 1)
            reward -= self.waiting_penalty * prio_weight_wait

        # 4) Neuen Call sampeln
        new_zone, new_prio = self._sample_call()
        if new_zone is not None:
            if not self.call_active:
                # neuer Call wird aktiv
                self.call_active = True
                self.call_zone = new_zone
                self.call_priority = new_prio
            else:
                # Call geht "verloren" => hohe Strafe (System überlastet)
                prio_weight_new = float(new_prio + 1)
                reward -= self.lost_call_penalty * prio_weight_new

        # 5) kleine Zeitschritt-Strafe
        reward -= self.timestep_penalty

        terminated = (self.step_count >= self.max_steps)
        truncated = False

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info


    def render(self):
        if self.render_mode != "human":
            return
        print(
            f"t={self.step_count} | "
            f"amb0: pos={self.amb0_pos}, busy={self.amb0_busy} | "
            f"amb1: pos={self.amb1_pos}, busy={self.amb1_busy} | "
            f"call: active={self.call_active}, zone={self.call_zone}, prio={self.call_priority}"
        )
