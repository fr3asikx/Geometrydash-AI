# Geometrydash-AI

Dieses Projekt implementiert eine vereinfachte Reinforcement-Learning-Umgebung, die sich am Spiel *Geometry Dash* orientiert. Ein Deep-Q-Network-Agent wird trainiert, um Hindernissen auszuweichen und ein Level zu vervollständigen. Für jedes Level wird automatisch ein eigenes Profil angelegt, in dem Fortschritte gespeichert werden. Während des Trainings erscheinen zwei Fenster: Eines zeigt den aktuellen Zustand des Levels und die Aktionen des Agenten, ein weiteres visualisiert den Trainingsfortschritt als Graph.

## Features

- **Level-Profile**: Persistente Profile pro Level, die automatisch erstellt und aktualisiert werden.
- **Neurales Netz**: Ein DQN-Agent mit Erfahrungsspeicher und Target-Netzwerk.
- **Reward-System**: Belohnungen basieren auf Fortschritt, Kollisionsvermeidung und erfolgreichen Abschlüssen.
- **Visualisierung**: Zwei Matplotlib-Fenster zeigen die Spielszene sowie Verlauf von Belohnungen und Verlusten.

## Voraussetzungen

Installiere die Abhängigkeiten (idealerweise in einer virtuellen Umgebung):

```bash
pip install -r requirements.txt
```

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
