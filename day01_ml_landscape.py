"""
day01_ml_landscape.py
Goal: Practice identifying Task (T), Experience (E), and Performance
measure (P) for real ML systems, per Tom Mitchell's definition.
"""

# A small "database" of ML systems you interact with regularly.
# Fill this in with YOUR OWN 5 examples before moving on — don't just
# copy these; the point is to practice the T/E/P framing yourself.
ml_systems = [
    {
        "name": "Email spam filter",
        "task": "Classify an incoming email as spam or not spam",
        "experience": "Historical emails labeled spam/ham by users",
        "performance_measure": "Accuracy (fraction of emails correctly classified)",
    },
    {
        "name": "Netflix recommendations",
        "task": "Rank which shows to display to a given user",
        "experience": "Past viewing history and ratings from many users",
        "performance_measure": "Click-through rate or watch time after recommendation",
    }
]

def print_ml_system_analysis(system: dict) -> None:
    """Print a system's T/E/P breakdown in a readable format."""
    print(f"\n=== {system['name']} ===")
    print(f"  Task (T):                {system['task']}")
    print(f"  Experience (E):          {system['experience']}")
    print(f"  Performance measure (P): {system['performance_measure']}")

if __name__ == "__main__":
    for system in ml_systems:
        print_ml_system_analysis(system)