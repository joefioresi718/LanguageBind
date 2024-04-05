import pandas as pd

# Load the CSV data into a DataFrame
df = pd.read_csv('CLIPblind_pairs_hmdb_vitl.csv')

# Calculate the preference score for each row
df['preference_score'] = df['clip_similarity'] + (1 - df['video_similarity'])

# Create a unique list of videos
videos = pd.unique(df[['video1', 'video2']].values.ravel('K'))

# List to store DataFrames for each best match
dfs_list = []

# Track videos already matched
matched_videos = set()

for video in videos:
    if video not in matched_videos:
        # Filter pairs that include the current video, not yet matched
        possible_pairs = df[((df['video1'] == video) | (df['video2'] == video)) & (~df['video1'].isin(matched_videos)) & (~df['video2'].isin(matched_videos))]
        
        # Sort by preference_score to get the best match
        best_pair = possible_pairs.sort_values(by='preference_score', ascending=False).head(1)
        
        if not best_pair.empty:
            # Add the best pair DataFrame to the list
            dfs_list.append(best_pair)
            # Update the set of matched videos
            matched_videos.update(best_pair['video1'].tolist())
            matched_videos.update(best_pair['video2'].tolist())

# Concatenate all the best match DataFrames together
best_matches_df = pd.concat(dfs_list, ignore_index=True)

# Save the refined DataFrame to a new CSV file
best_matches_df.to_csv('CLIPblind_pairs_hmdb_vitl_final.csv', index=False)
