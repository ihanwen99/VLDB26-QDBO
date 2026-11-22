import json
import os


def load_statistics(sql_folder_path):
    cardinalities_file_path = "{}/cardinalities.json".format(sql_folder_path)
    selectivities_file_path = "{}/selectivities.json".format(sql_folder_path)
    with open(cardinalities_file_path, "r") as file:
        cardinalities_content = json.load(file)

    with open(selectivities_file_path, "r") as file:
        selectivities_content = json.load(file)
    return cardinalities_content, selectivities_content


def parse_predicate_selectivities(sel):
    pred = []
    pred_sel = []
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            if sel[i][j] != 1:
                pred.append((i, j))
                pred_sel.append(sel[i][j])
    return pred, pred_sel


def read_query_sql(sql_folder_path: str):
    sql_files = [f for f in os.listdir(sql_folder_path) if f.endswith(".sql")]
    sql_file_path = os.path.join(sql_folder_path, sql_files[0])
    with open(sql_file_path, "r", encoding="utf-8") as file:
        query = file.read()

    return query


def list_folders_by_number(directory):
    all_items = os.listdir(directory)
    folders = [
        item for item in all_items if os.path.isdir(os.path.join(directory, item))
    ]
    sorted_folders = sorted(
        folders,
        key=lambda x: (
            int("".join(filter(str.isdigit, x)))
            if any(c.isdigit() for c in x)
            else float("inf")
        ),
    )
    return [os.path.join(directory, folder) for folder in sorted_folders]
