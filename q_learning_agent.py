import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from rtw_verteilung import EmergencyDispatchEnv


def obs_to_index(obs, env: EmergencyDispatchEnv) -> int:
    """
    MultiDiscrete-Observation -> eindeutiger 1D-Index für die Q-Table.

    Die Struktur der Observation ist (aus rtw_verteilung.py):
        [amb0_pos, amb0_busy, amb1_pos, amb1_busy, call_zone, call_priority]

    Mit den nvec-Komponenten:
        n0 = num_zones
        n1 = max_busy + 1
        n2 = num_zones
        n3 = max_busy + 1
        n4 = num_zones + 1
        n5 = 4
    """
    n0 = env.num_zones
    n1 = env.max_busy + 1
    n2 = env.num_zones
    n3 = env.max_busy + 1
    n4 = env.num_zones + 1
    n5 = 4

    a0, a1, a2, a3, a4, a5 = obs

    index = (
        a0
        + n0 * (
            a1
            + n1 * (
                a2
                + n2 * (
                    a3
                    + n3 * (
                        a4
                        + n4 * a5
                    )
                )
            )
        )
    )
    return int(index)

def shaping_potential(obs, env: EmergencyDispatchEnv) -> float:
    """
    Potentialfunktion für Reward Shaping.

    Idee:
    - Wenn kein aktiver Call vorhanden ist (call_zone == num_zones): 0
    - Sonst:
      - Den minimalen Abstand eines freien RTW zur Call-Zone
      - Höhere Priorität = höheres Gewicht

    Rückgabe:
    - Je kleiner der Abstand, desto höher (besser) die Potentialfunktion
    """
    a0_pos, a0_busy, a1_pos, a1_busy, call_zone, call_priority = obs

    # Fall: kein aktiver Notruf
    if call_zone == env.num_zones:
        return 0.0

    distances = []

    # Nur freie RTWs berücksichtigen
    if a0_busy == 0:
        distances.append(abs(a0_pos - call_zone))
    if a1_busy == 0:
        distances.append(abs(a1_pos - call_zone))

    if not distances:
        # Kein freier RTW verfügbar -> kein Shaping
        return 0.0

    best_dist = min(distances)

    # Gewichtung nach Priorität 
    priority_weight = 1.0 + float(call_priority)  

    # Kleinere Distanz => höherer Wert
    return -best_dist * priority_weight

def train_q_learning(
    num_episodes: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int = 42,
    ax=None,                  
    line=None,                
    glow=None,                
    label: str = "",          
    live_plot: bool = False,  
):
    np.random.seed(seed)
    random.seed(seed)

    try:
        env = EmergencyDispatchEnv(render_mode=None, seed=seed)
    except TypeError:
        env = EmergencyDispatchEnv(render_mode=None)

    obs_space_n = int(np.prod(env.observation_space.nvec))
    action_space_n = env.action_space.n
    Q = np.zeros((obs_space_n, action_space_n))

    epsilon = epsilon_start
    rewards_history = []

    for episode in range(num_episodes):
        try:
            obs, info = env.reset(seed=seed + episode)
        except TypeError:
            obs, info = env.reset()

        state = obs_to_index(obs, env)
        done = False
        total_reward = 0.0

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = obs_to_index(next_obs, env)

            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (
                reward + gamma * best_next * (not terminated) - Q[state, action]
            )

            state = next_state
            total_reward += reward
            done = terminated or truncated

        rewards_history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Logging + Live-Plot Update
        if (episode + 1) % 100 == 0:
            avg_last_100 = float(np.mean(rewards_history[-100:]))
            print(
                f"Episode {episode + 1}, "
                f"total_reward={total_reward:.2f}, "
                f"avg_last_100={avg_last_100:.2f}, "
                f"epsilon={epsilon:.3f}"
            )

            if live_plot and ax is not None and line is not None and len(rewards_history) >= 100:
                window = 100
                moving_avg = np.convolve(
                    rewards_history,
                    np.ones(window) / window,
                    mode="valid"
                )
                x = np.arange(window, window + len(moving_avg))

                line.set_data(x, moving_avg)
                if glow is not None:
                    glow.set_data(x, moving_avg)

                ax.set_xlim(0, num_episodes)
                ax.set_ylim(np.min(moving_avg) - 50, np.max(moving_avg) + 50)
                ax.figure.canvas.draw_idle()
                plt.pause(0.01)

    return Q, rewards_history, env

def evaluate_policy(env: EmergencyDispatchEnv, Q: np.ndarray, num_episodes: int = 200):
    """
    Evaluierung der gelernten Policy ohne Exploration (epsilon = 0).

    - nimmt immer argmax(Q[s])
    - gibt Durchschnittsreward über num_episodes zurück
    """
    total = 0.0

    for _ in range(num_episodes):
        obs, info = env.reset()
        state = obs_to_index(obs, env)
        done = False
        ep_reward = 0.0

        while not done:
            action = int(np.argmax(Q[state]))  # reine greedy-Policy
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = obs_to_index(next_obs, env)

            ep_reward += reward
            state = next_state
            done = terminated or truncated

        total += ep_reward

    avg_reward = total / num_episodes
    print(f"[EVAL] Greedy-Policy: Durchschnittsreward über {num_episodes} Episoden: {avg_reward:.2f}")
    return avg_reward


def evaluate_random_policy(env: EmergencyDispatchEnv, num_episodes: int = 200):
    """
    Vergleich: komplett zufällige Policy.

    - wählt in jedem Schritt random eine Action
    - zeigt dir, wie 'dumm' Zufall ist im Vergleich zu deinem gelernten Agenten
    """
    total = 0.0

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        total += ep_reward

    avg_reward = total / num_episodes
    print(f"[EVAL] Random-Policy: Durchschnittsreward über {num_episodes} Episoden: {avg_reward:.2f}")
    return avg_reward

def evaluate_greedy_and_random(
    Q: np.ndarray,
    num_eval_episodes: int = 200,
    seed: int = 0,
):
    """
    Evaluiert:
    - Greedy-Policy (immer argmax_a Q(s,a))
    - Random-Policy (rein zufällige Aktionen)

    und gibt die durchschnittlichen Episodenrewards zurück.
    """
    # Greedy-Policy
    np.random.seed(seed)
    random.seed(seed)
    env_g = EmergencyDispatchEnv(render_mode=None)
    avg_greedy = evaluate_policy(env_g, Q, num_episodes=num_eval_episodes)

    # Random-Policy
    np.random.seed(seed)
    random.seed(seed)
    env_r = EmergencyDispatchEnv(render_mode=None)
    avg_random = evaluate_random_policy(env_r, num_episodes=num_eval_episodes)

    return avg_greedy, avg_random


def multi_seed_experiment(
    seeds,
    num_episodes: int = 2000,
    num_eval_episodes: int = 200,
    live_plot: bool = True,
):
    results = []
    seed_lines = {}   # für Plot-Linien pro Seed
    seed_scores = {}  # für beste Seed-Auswahl
    greedy_values = []
    random_values = []

    # ====== EIN Fenster für alle Seeds ======
    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        # --- Fullscreen / Maximized Window ---
        manager = plt.get_current_fig_manager()

        try:
            # macOS (TkAgg)
            manager.window.state("zoomed")
        except Exception:
            try:
            # Qt backend (Windows / Linux)
                manager.window.showMaximized()
            except Exception:
                try:
                    # Fallback: Fullscreen mode
                    manager.full_screen_toggle()
                except Exception:
                    pass
# ------------------------------------

        fig.patch.set_facecolor("#0b0f19")
        ax.set_facecolor("#0f1629")
        ax.set_title("RTW Dispatch – Live Lernkurven (Moving Avg 100)", fontsize=14, pad=12, color="#cfd8ff")
        ax.set_xlabel("Episode", color="#cfd8ff")
        ax.set_ylabel("Ø Reward (letzte 100 Episoden)", color="#cfd8ff")
        ax.grid(True, alpha=0.15)
        for spine in ax.spines.values():
            spine.set_color("#2a3553")
        ax.tick_params(colors="#cfd8ff")
    else:
        ax = None

    # Farben für Seeds im Plot
    palette = ["#00e5ff", "#ff4fd8", "#7c5cff", "#2bff88", "#ffb703"]

    for i, s in enumerate(seeds):
        print(f"\n=== Training mit Seed {s} ===")

        # pro Seed neue Linie im gleichen Plot
        if live_plot:
            c = palette[i % len(palette)]
            line, = ax.plot([], [], color=c, linewidth=2.6, label=f"Seed {s}")
            glow, = ax.plot([], [], color=c, linewidth=6, alpha=0.12)
            seed_lines[s] = (line, glow)
            ax.legend(frameon=False, ncol=3, fontsize=9, title_fontsize=10, labelcolor="linecolor")
        else:
            line = glow = None

        Q, rewards_history, env = train_q_learning(
            num_episodes=num_episodes,
            seed=s,
            ax=ax,
            line=line,
            glow=glow,
            label=f"Seed {s}",
            live_plot=live_plot,
        )

        avg_greedy, avg_random = evaluate_greedy_and_random(
            Q,
            num_eval_episodes=num_eval_episodes,
            seed=1000 + s,
        )
        seed_scores[s] = avg_greedy

        print(
            f"[MULTI-SEED] Seed={s} | "
            f"Greedy-Policy avg_reward={avg_greedy:.2f} | "
            f"Random-Policy avg_reward={avg_random:.2f}"
        )

        results.append((s, avg_greedy, avg_random))
        greedy_values.append(avg_greedy)
        random_values.append(avg_random)
        
    # ===== Best Seed bestimmen =====
    best_seed = max(seed_scores, key=seed_scores.get)
    
    for s, (line, glow) in seed_lines.items():
        if s == best_seed:
            # BEST SEED – Highlight
            line.set_linewidth(4.5)
            line.set_zorder(10)
            glow.set_alpha(0.25)
        else:
            # Andere Seeds dezent darstellen
            line.set_alpha(0.35)
            glow.set_alpha(0.05)
    
    # ===== Best Seed Label im Plot =====
    ax.text(
        0.01, 0.98,
        f"Best Seed: {best_seed}",
        transform=ax.transAxes,
        fontsize=11,
        color="#cfd8ff",
        fontweight="bold",
        verticalalignment="top",
    )

    # Zusammenfassung
    greedy_values = np.array(greedy_values, dtype=float)
    random_values = np.array(random_values, dtype=float)

    mean_greedy = float(greedy_values.mean())
    std_greedy = float(greedy_values.std())
    mean_random = float(random_values.mean())
    std_random = float(random_values.std())

    print("\n=== Zusammenfassung Multi-Seed-Experiment ===")
    print(f"Greedy-Policy:  Mittelwert = {mean_greedy:.2f}, Std-Abw = {std_greedy:.2f}")
    print(f"Random-Policy:  Mittelwert = {mean_random:.2f}, Std-Abw = {std_random:.2f}")

    if live_plot:
        plt.ioff()
        plt.show()

    return results

if __name__ == "__main__":
    """
    Hier kann man auswählen, ob man ein 
    einzelnes Training oder ein Multi-Seed-Experiment 
    durchführen möchte.
    """
    # # # Einzelnes Training und Evaluierung
    # # Q, rewards_history, env = train_q_learning(num_episodes=2000)

    # # print("\nTraining fertig.\n")

    # # # Evaluierung: gelernte Policy (greedy)
    # # evaluate_policy(env, Q, num_episodes=200)

    # # # Evaluierung: Random Policy zum Vergleich
    # # evaluate_random_policy(env, num_episodes=200)
    
    # Multi-Seed-Experiment
    seeds = [1, 7, 42, 123, 999]
    multi_seed_experiment(seeds, num_episodes=2000, num_eval_episodes=200)
