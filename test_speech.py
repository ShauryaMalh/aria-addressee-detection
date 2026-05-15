import pandas as pd

path = "/homes/iws/shaurm/projectaria_sandbox/projectaria_tools/data/gen1/aria_everyday_activities_test_data/speech.csv"
df = pd.read_csv(path)

print("Columns:", list(df.columns))
print(f"Number of word rows: {len(df)}\n")

# Cluster consecutive words into utterances (gap < 1 second = same utterance)
GAP_NS = 1_000_000_000  # 1 second in nanoseconds

utterances = []
current = [df.iloc[0]]
for i in range(1, len(df)):
    prev_end = df.iloc[i-1]["endTime_ns"]
    curr_start = df.iloc[i]["startTime_ns"]
    if curr_start - prev_end < GAP_NS:
        current.append(df.iloc[i])
    else:
        utterances.append(current)
        current = [df.iloc[i]]
utterances.append(current)

print(f"Found {len(utterances)} utterances:\n")
for u in utterances:
    start_s = u[0]["startTime_ns"] / 1e9
    end_s = u[-1]["endTime_ns"] / 1e9
    text = " ".join(w["written"] for w in u)
    print(f"  [{start_s:.2f}s - {end_s:.2f}s]  {text}")
