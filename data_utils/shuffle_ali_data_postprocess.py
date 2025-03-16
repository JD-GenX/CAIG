import json
import os
import random
import cv2
import numpy as np
from tqdm import tqdm

prompt_pool=[
"<image>\nComparing the left part and right part of this image, which part is more suitable for the product '{}'?",
"<image>\nThe left and right part of this image is one advertising image for the product '{}', respectively, which is preferred by the user?",
"<image>\nWhich part will bring more click-through rate in this image for product '{}' ?",
"<image>\nBetween the left and right sections of this image, which one is more appropriate for showcasing the product '{}'?",
"<image>\nConsidering the left and right halves of this image, which side better represents the product '{}'?",
"<image>\nWhich side of this image, left or right, is more effective for advertising the product '{}'?",
"<image>\nFor the product '{}', which part of the image, left or right, is more appealing?",
"<image>\nWhen looking at the left and right portions of this image, which part is more suitable for promoting the product '{}'?",
"<image>\nIn this image, which side, left or right, is preferred by users for the product '{}'?",
"<image>\nWhich half of this image, left or right, is more likely to attract user preference for the product '{}'?",
"<image>\nWhich section of this image, left or right, is expected to generate a higher click-through rate for the product '{}'?",
"<image>\nFor the product '{}', which side of the image do users find more engaging, left or right?",
"<image>\nBetween the left and right sides of this image, which one is anticipated to drive more clicks for the product '{}'?",
]

nocap_prompt_pool=[
"<image>\nComparing the left part and right part of this image, which part is more suitable for advertising?",
"<image>\nThe left and right part of this image is one advertising image for a product, respectively, which is preferred by the user?",
"<image>\nWhich part will bring more click-through rate in this image?",
"<image>\nWhich side of this image, left or right, is more effective for advertising the product?",
"<image>\nWhich part of the image, left or right, is more appealing?",
"<image>\nIn this image, which side, left or right, is preferred by users?",
]

additional_msg_prompt =[
" More product information: the click-through rate of the product is {}; the coarse-to-fine categories of the product are {}, {} and {}; the mobile price of the product is {}; the common price of the product is {}; the WeChat price of the product is {}; the amount of the good comments is {}; the amount of the order in recent one year is {}. ",

" Additional information: the click-through rate of the product is {}; the coarse-to-fine categories of the product are {}, {} and {}; the mobile price of the product is {}; the common price of the product is {}; the WeChat price of the product is {}; the amount of the good comments is {}; the amount of the order in recent one year is {}. ",

" Product stats: click-through rate is {}; categorized as {}, {} and {}; mobile price: {}; common price: {}; WeChat price: {}; number of good comments: {}; orders in the last year: {}. ",

" Detailed product info: click-through rate: {}; categories include {}, {} and {}; mobile price: {}; regular price: {}; WeChat price: {}; number of positive reviews: {}; orders in the past year: {}. ", 

" Product metrics: click-through rate: {}; categories: {}, {} and {}; mobile price: {}; standard price: {}; WeChat price: {}; positive comments: {}; yearly orders: {}. ",
]

left_prompt =[
"Left",
"The left part is better",
"Users will prefer to the left side.",
"The left side is more suitable.",
"Users favor the left side.",
"The left part is more attractive.",
"The left half will bring higher click-through rate.",
"The left part will attract more users.",
]

right_prompt = [
    "Right",
    "The right section is superior.",
    "Users will be drawn to the right side.",
    "The right side is more appropriate.",
    "Users prefer the right side.",
    "The right portion is more appealing.",
    "Right half will bring a higher click-through rate.",
    "The left part will attract more customers."
]

def calculate_ctr(item):
    if item["exp_impression_cnt"] == 0:
        return 0
    return item["exp_click_cnt"] / item["exp_impression_cnt"]


def process_data(data):
    result = {}
    new_items = []
    result_key = 0
    for key, items in tqdm(data.items(), total=len(data.items())):
        if not items:
            continue

        items_with_ctr = [(item, calculate_ctr(item)) for item in items]

        sorted_items = sorted(items_with_ctr, key=lambda x: x[1])
        for i in range(len(sorted_items)):
            for j in range(i + 1, len(sorted_items)):
                item1, ctr1 = sorted_items[i]
                item2, ctr2 = sorted_items[j]

                if ctr1 == 0 or ctr2 == 0:
                    continue

                if ctr2 / ctr1 <= 1 + max_ctr_gap * 0.01 and ctr2 / ctr1 >= 1 + min_ctr_gap * 0.01:
                    image_2 = cv2.imread(os.path.join("dataset/ali_dataset/images",os.path.basename(item2["creative_image_hash"])), cv2.IMREAD_UNCHANGED)
                    image_2 = cv2.resize(image_2, dsize=(512, 512))
                    image_1 = cv2.imread(os.path.join("dataset/ali_dataset/images",os.path.basename(item1["creative_image_hash"])), cv2.IMREAD_UNCHANGED)
                    image_1 = cv2.resize(image_1, dsize=(512, 512))

                    if random.choice([True, False]):
                        new_image = np.concatenate([image_1, image_2], axis=1)
                        correct_side = "right"
                    else:
                        new_image = np.concatenate([image_2, image_1], axis=1)
                        correct_side = "left"

                    save_path = os.path.join(save_dir, "{}_{}.png".format(key, result_key))
                    cv2.imwrite(save_path, new_image)

                    new_item = {
                        "id": "{}_{}".format(key, result_key),
                        "image": save_path,
                        "left_ctr": ctr2 if correct_side == "left" else ctr1,
                        "right_ctr": ctr1 if correct_side == "left" else ctr2,
                        "conversations": [{
                            "from": "human",
                            "value": random.choice(nocap_prompt_pool)
                        },
                        {
                            "from": "gpt",
                            "value": random.choice(left_prompt) if correct_side == "left" else random.choice(right_prompt)
                        },],
                        "target": correct_side
                    }
                    new_items.append(new_item)
                    result_key += 1
    return new_items


def read_caption_json_file(file_path):
    creative_dict = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            creative_id = data["creative_id"]
            creative_dict[creative_id] = data

    return creative_dict


save_dir = "ali_dataset/"
os.makedirs(save_dir, exist_ok=True)

min_ctr_gap = 5
max_ctr_gap = 20
min_exp_impression_cnt = 1000

with open("json/ali_filter_1000_5_20.json", "r") as f:
    input_data = json.load(f)

output_data = process_data(input_data)
print(len(output_data))

output_json_path = f"json/ali_dataset_filter_{min_exp_impression_cnt}_{min_ctr_gap}_{max_ctr_gap}.json"
with open(output_json_path, "w") as f:

    json.dump(output_data, f, indent=2)

print(f"All images done. JSON File is Saved to {output_json_path}")
