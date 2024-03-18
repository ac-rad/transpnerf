

import sys
import os
import pandas as pd

def jsons_to_excel(input_folder, output_excel):
    # Get a list of JSON files in the input folder
    json_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.json')]

    df = pd.DataFrame()

    for json_file in json_files:
        with open(json_file) as f:
            json_data = json.load(f)
        results = json_data["results"]

        if results:
            file_name = os.path.basename(json_file)
            df[file_name] = results

    df.to_excel(output_excel, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: get_eval_results.py <input_folder> <output_excel>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_excel = sys.argv[2]
    jsons_to_excel(input_folder, output_excel)



# to add:
# chamfer distance point clouds
# l2 norm depth maps