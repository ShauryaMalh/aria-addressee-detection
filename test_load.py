from projectaria_tools.core import data_provider

vrs_path = "/homes/iws/shaurm/projectaria_sandbox/projectaria_tools/data/gen1/aria_everyday_activities_test_data/recording.vrs"
provider = data_provider.create_vrs_data_provider(vrs_path)

print("VRS file opened successfully\n")

streams = provider.get_all_streams()
print(f"Number of streams: {len(streams)}\n")

for stream_id in streams:
    label = provider.get_label_from_stream_id(stream_id)
    num_data = provider.get_num_data(stream_id)
    print(f"  {stream_id}  |  {label:30s}  |  {num_data} samples")
