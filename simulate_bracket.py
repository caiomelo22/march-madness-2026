import pandas as pd

# Load data
sub = pd.read_csv("./kaggle/working/submission.csv")
seeds = pd.read_csv(
    "./kaggle/input/competitions/march-machine-learning-mania-2026/MNCAATourneySeeds.csv"
)
teams = pd.read_csv(
    "./kaggle/input/competitions/march-machine-learning-mania-2026/MTeams.csv"
)

seeds_2026 = seeds[seeds["Season"] == 2026].copy()
seeds_2026 = seeds_2026.merge(teams[["TeamID", "TeamName"]], on="TeamID")

# Build lookup: (teamA_id, teamB_id) -> prob teamA wins (A < B always in dataset)
sub_dict = {}
for _, row in sub.iterrows():
    sub_dict[(int(row["TeamA"]), int(row["TeamB"]))] = row["Pred"]


def get_pred(team_a_id, team_b_id):
    """Returns probability that team_a_id beats team_b_id"""
    a, b = min(team_a_id, team_b_id), max(team_a_id, team_b_id)
    key = (a, b)
    if key in sub_dict:
        prob = sub_dict[key]
        return prob if team_a_id == a else 1.0 - prob
    return 0.5  # fallback


# Create team lookup
team_info = seeds_2026.set_index("TeamID")[["Seed", "TeamName"]].to_dict("index")


def show(team_id):
    info = team_info.get(team_id, {})
    return f"{info.get('Seed','?'):5s} {info.get('TeamName','?')} ({team_id})"


def simulate_game(id1, id2, verbose=True):
    prob = get_pred(id1, id2)
    winner = id1 if prob >= 0.5 else id2
    if verbose:
        prob_display = prob if winner == id1 else 1.0 - prob
        print(
            f"  {show(id1)} vs {show(id2)}  -> pred={prob:.3f} -> WINNER: {team_info[winner]['TeamName']} ({prob_display:.1%})"
        )
    return winner


# Create seed->teamID dict
seed_to_id = dict(zip(seeds_2026["Seed"], seeds_2026["TeamID"]))

# Play-in games
print("=== PLAY-IN GAMES ===")
x16_winner = simulate_game(seed_to_id["X16a"], seed_to_id["X16b"])
y16_winner = simulate_game(seed_to_id["Y16a"], seed_to_id["Y16b"])
y11_winner = simulate_game(seed_to_id["Y11a"], seed_to_id["Y11b"])
z11_winner = simulate_game(seed_to_id["Z11a"], seed_to_id["Z11b"])
seed_to_id["X16"] = x16_winner
seed_to_id["Y16"] = y16_winner
seed_to_id["Y11"] = y11_winner
seed_to_id["Z11"] = z11_winner

regions = {
    "W (West)": [
        "W01",
        "W16",
        "W08",
        "W09",
        "W05",
        "W12",
        "W04",
        "W13",
        "W06",
        "W11",
        "W03",
        "W14",
        "W07",
        "W10",
        "W02",
        "W15",
    ],
    "X (East)": [
        "X01",
        "X16",
        "X08",
        "X09",
        "X05",
        "X12",
        "X04",
        "X13",
        "X06",
        "X11",
        "X03",
        "X14",
        "X07",
        "X10",
        "X02",
        "X15",
    ],
    "Y (South)": [
        "Y01",
        "Y16",
        "Y08",
        "Y09",
        "Y05",
        "Y12",
        "Y04",
        "Y13",
        "Y06",
        "Y11",
        "Y03",
        "Y14",
        "Y07",
        "Y10",
        "Y02",
        "Y15",
    ],
    "Z (Midwest)": [
        "Z01",
        "Z16",
        "Z08",
        "Z09",
        "Z05",
        "Z12",
        "Z04",
        "Z13",
        "Z06",
        "Z11",
        "Z03",
        "Z14",
        "Z07",
        "Z10",
        "Z02",
        "Z15",
    ],
}

round_names = {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16", 4: "Elite 8"}

final_four = []
for region_name, bracket_seeds in regions.items():
    print(f'\n{"="*60}')
    print(f"  {region_name} REGION")
    print(f'{"="*60}')
    current_round = [seed_to_id[s] for s in bracket_seeds]

    for round_num in range(1, 5):
        print(f"\n-- {round_names[round_num]} --")
        next_round = []
        for i in range(0, len(current_round), 2):
            winner = simulate_game(current_round[i], current_round[i + 1])
            next_round.append(winner)
        current_round = next_round

    final_four.append(current_round[0])
    print(f'\n>> Region Winner: {team_info[current_round[0]]["TeamName"]}')

print(f'\n{"="*60}')
print("  FINAL FOUR")
print(f'{"="*60}')
print("\n-- Semifinal 1 (W vs X) --")
sf1_winner = simulate_game(final_four[0], final_four[1])
print("\n-- Semifinal 2 (Y vs Z) --")
sf2_winner = simulate_game(final_four[2], final_four[3])

print(f'\n{"="*60}')
print("  CHAMPIONSHIP GAME")
print(f'{"="*60}')
champion = simulate_game(sf1_winner, sf2_winner)
print(f'\n*** CHAMPION: {team_info[champion]["TeamName"]} ***')
