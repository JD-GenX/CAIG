import json


def calculate_ctr(item):
    if item["exp_impression_cnt"] == 0:
        return 0
    return item["exp_click_cnt"] / item["exp_impression_cnt"]


def process_data(data):
    result = {}
    result_key = 0
    for key, items in data.items():
        if not items:
            continue

        filtered_items = [item for item in items if item["exp_impression_cnt"] > min_exp_impression_cnt]

        if len(filtered_items) < 2:
            continue

        items_with_ctr = [(item, calculate_ctr(item)) for item in filtered_items]

        sorted_items = sorted(items_with_ctr, key=lambda x: x[1])

        for i in range(len(sorted_items)):
            for j in range(i + 1, len(sorted_items)):
                item1, ctr1 = sorted_items[i]
                item2, ctr2 = sorted_items[j]

                if ctr1 == 0 or ctr2 == 0:
                    continue
                if ctr2 / ctr1 <= 1 + max_ctr_gap * 0.01 and ctr2 / ctr1 >= 1 + min_ctr_gap * 0.01:
                    result[result_key] = [item2, item1]
                    result_key += 1

    return result


def read_caption_json_file(file_path):
    creative_dict = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            creative_id = data["creative_id"]
            creative_dict[creative_id] = data
    return creative_dict


min_ctr_gap = 5
max_ctr_gap = 20
min_exp_impression_cnt = 1000
file_path = "shopee_data/caption_public.json"
creative_caption_dict = read_caption_json_file(file_path)

with open("shopee_data/test.json", "r") as f:
    input_data = json.load(f)

output_data = process_data(input_data)
print(len(output_data))
output_json_path = f"json/ali_test_filter_{min_exp_impression_cnt}_{min_ctr_gap}_{max_ctr_gap}.json"
with open(output_json_path, "w") as f:

    json.dump(output_data, f, indent=2)

print(f"JSON File is Saved to {output_json_path}")
