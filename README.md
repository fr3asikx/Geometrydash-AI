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

# Bildschirmbereich wählen (x1, y1, x2, y2) – an das eigene Setup anpassen.
capture = ScreenCapture(CaptureConfig(region=(0, 0, 1280, 720), downscale=2))
estimator = StateEstimator()
controller = InputController()
interface = GeometryDashScreenInterface(capture, estimator, controller)
env = RealGeometryDashEnv(interface)

agent = DQNAgent(state_dim=env.reset().shape[0], action_dim=len(env.ACTIONS))
trainer = Trainer(env, agent, level=demo_level("live"), config=TrainingConfig(episodes=100))
trainer.train()
```

Während das Level aktiv läuft, liest das Interface kontinuierlich den Bildschirm aus, schätzt den Zustand und sendet Sprünge/Klicks entsprechend der Aktionen des Agenten. Die Trainingsvisualisierung bleibt identisch, kann aber bei Bedarf deaktiviert werden (`--no-visualization`).
