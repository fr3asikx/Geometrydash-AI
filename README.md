# Geometrydash-AI

Dieses Projekt implementiert eine vereinfachte Reinforcement-Learning-Umgebung, die sich am Spiel *Geometry Dash* orientiert. Ein Deep-Q-Network-Agent wird trainiert, um Hindernissen auszuweichen und ein Level zu vervollständigen. Für jedes Level wird automatisch ein eigenes Profil angelegt, in dem Fortschritte gespeichert werden. Während des Trainings erscheinen zwei Fenster: Eines zeigt den aktuellen Zustand des Levels und die Aktionen des Agenten, ein weiteres visualisiert den Trainingsfortschritt als Graph.

## Features

- **Level-Profile**: Persistente Profile pro Level, die automatisch erstellt und aktualisiert werden.
- **Neurales Netz**: Ein DQN-Agent mit Erfahrungsspeicher und Target-Netzwerk.
- **Reward-System**: Belohnungen basieren auf Fortschritt, Kollisionsvermeidung und erfolgreichen Abschlüssen.
- **Visualisierung**: Zwei Matplotlib-Fenster zeigen die Spielszene sowie Verlauf von Belohnungen und Verlusten.
- **Live-Interface**: Bildschirmaufnahme (dxcam/mss), Zustandsabschätzung per Computer Vision und automatische Eingaben über SendInput bzw. PyAutoGUI/PyDirectInput.

## Voraussetzungen

Installiere die Abhängigkeiten (idealerweise in einer virtuellen Umgebung):

```bash
pip install -r requirements.txt
```

### Zusätzliche Hinweise für das Live-Interface

- Unter Windows wird `dxcam` automatisch verwendet, ansonsten `mss`.
- Für zuverlässige Eingaben in DirectX-Spielen wird `pydirectinput` empfohlen. Alternativ fällt das System auf `pyautogui` zurück.
- Passe bei Bedarf die Farbschwellenwerte in `geometrydash_ai/game_interface.py` (`EstimatorConfig`) an, damit Spieler und Hindernisse korrekt erkannt werden.

## Training starten

Starte das Training mit:

```bash
python main.py --episodes 200
```

Optionen:

- `--level`: Name des Levels (Standard: `training_ground`).
- `--episodes`: Anzahl der Trainings-Episoden.
- `--no-visualization`: Deaktiviert die Matplotlib-Fenster.
- `--model-dir`: Ordner für gespeicherte Modelle.
- `--profiles-dir`: Ordner für Level-Profile.
- `--device`: Torch-Device (`cpu` oder `cuda`).

Beim Training werden Modelle regelmäßig im angegebenen Model-Ordner abgelegt und Level-Profile im Profil-Ordner gespeichert.

## Live-Training gegen das echte Spiel

Für Versuche direkt im laufenden Geometry-Dash-Client kann der simulierte `GeometryDashEnv` durch die Echtzeit-Variante ersetzt werden:

```python
from geometrydash_ai import (
    CaptureConfig,
    GeometryDashScreenInterface,
    InputController,
    RealGeometryDashEnv,
    ScreenCapture,
    StateEstimator,
    Trainer,
    TrainingConfig,
    DQNAgent,
)

# Standardmäßig wird automatisch das Fenster der `GeometryDash.exe` erfasst.
capture = ScreenCapture(CaptureConfig(downscale=2))
estimator = StateEstimator()
controller = InputController()
interface = GeometryDashScreenInterface(capture, estimator, controller)
env = RealGeometryDashEnv(interface)

agent = DQNAgent(state_dim=env.reset().shape[0], action_dim=len(env.ACTIONS))
trainer = Trainer(env, agent, level=demo_level("live"), config=TrainingConfig(episodes=100))
trainer.train()
```

Während das Level aktiv läuft, liest das Interface kontinuierlich das Fenster der `GeometryDash.exe` aus, schätzt den Zustand und sendet Sprünge/Klicks entsprechend der Aktionen des Agenten. Die Trainingsvisualisierung bleibt identisch, kann aber bei Bedarf deaktiviert werden (`--no-visualization`).

> **Hinweis:** Falls mehrere Fenster der `GeometryDash.exe` geöffnet sind oder ein anderer Pfad verwendet wird, kann der Prozessname über `CaptureConfig(process_name="EigenerName.exe")` angepasst werden. Bei Bedarf lässt sich die automatische Fenstersuche auch durch Angabe einer expliziten Region überschreiben.

## PPO (Sim-First + Transfer)

Neben dem DQN-Training auf aufgezeichneten Frames enthält das Projekt nun einen deterministischen Simulator (`GDSimEnv`), der eine Tilemap aus CSV-Dateien lädt. Darauf aufbauend können PPO-Policies mit `stable-baselines3` trainiert und anschließend in das echte Spiel übertragen werden.

### Setup

```bash
pip install -r requirements.txt
```

Der Simulator erwartet Level-Dateien im Ordner `levels/` (Beispiel: `level1.csv`). Ein Durchlauf entspricht einer 1D-Autoscroll-Strecke mit Achsparallelen Hindernissen.

### Training im Simulator

```bash
python -m geometrydash_ai.ppo.train_ppo --level levels/level1.csv --timesteps 5000000
```

Wichtige Parameter:

- `--dt`: Physik-Timestep (Standard `1/240 s`).
- `--n-envs`: Anzahl parallelisierter Simulatorinstanzen (SubprocVecEnv).
- `--tensorboard`: Optionaler Pfad für TensorBoard-Logs.

Eine verkürzte Variante für Tests steht als Skript bereit:

```bash
./scripts/run_train_ppo.sh
```

Während des Trainings werden Mittelwerte der Episoden-Rückgaben sowie der beste Fortschritt (0–100 %) im Terminal angezeigt. Das finale Modell wird standardmäßig unter `models/ppo_gd_level1.zip` gespeichert.

### Transfer ins echte Spiel

```bash
python -m geometrydash_ai.ppo.play_transfer --model models/ppo_gd_level1.zip --fps 240
```

Das Skript erzeugt einen Proxy-Zustand (identisch zum Simulator-Feature-Vektor) und sendet Space-Eingaben an die fokussierte `Geometry Dash`-Instanz. Für Experimente mit Computer-Vision kann `--vision` gesetzt werden, wodurch Screenshot-basierte Beobachtungen (`128×72`, Graustufen) erzeugt werden. Das Skript enthält einen einfachen Frame-Timer (Standard 240 FPS) sowie ein Debouncing, damit Tap-Aktionen nur einmal pro physikalischem Frame ausgelöst werden.

Schnellaufruf:

```bash
./scripts/run_play_transfer.sh
```

> **Praxis-Tipp:** Für reproduzierbare Ergebnisse empfiehlt sich das Aktivieren des Practice-Mode oder ein FPS-Lock auf 240 Hz, damit der reale Client mit der Simulator-Taktung synchron läuft.

> **Rechtlicher Hinweis:** Die Tools sind ausschließlich für Einzelspieler- und Offline-Experimente gedacht. Bitte keine kompetitiven Modi oder Online-Ranglisten beeinflussen.
