def clevr_rel_to_text(clevr_rel):

    # takes a list of 9 integers and returns a string
    colors_to_idx = {
        "gray": 0,
        "red": 1,
        "blue": 2,
        "green": 3,
        "brown": 4,
        "purple": 5,
        "cyan": 6,
        "yellow": 7,
        "none": 8,
    }
    shapes_to_idx = {"cube": 0, "sphere": 1, "cylinder": 2, "none": 3}
    materials_to_idx = {"rubber": 0, "metal": 1, "none": 2}
    sizes_to_idx = {"small": 0, "large": 1, "none": 2}
    relations_to_idx = {
        "left": 0,
        "right": 1,
        "front": 2,
        "behind": 3,
        "below": 4,
        "above": 5,
        "none": 6,
    }

    label_description = {
        "left": "to the left of",
        "right": "to the right of",
        "behind": "behind",
        "front": "in front of",
        "above": "above",
        "below": "below",
    }
    colors = list(colors_to_idx.keys())
    shapes = list(shapes_to_idx.keys())
    materials = list(materials_to_idx.keys())
    sizes = list(sizes_to_idx.keys())
    relations = list(relations_to_idx.keys())

    text_label = []
    for i in range(2):  # take the two objects (two of the objects...)
        shape, size, color, material, pos = clevr_rel[i * 5 : i * 5 + 5]
        obj = " ".join(
            [sizes[size], colors[color], materials[material], shapes[shape]]
        ).strip()
        text_label.append(obj)
    relation = relations[clevr_rel[-1]]  # there's only one relation per image
    if "none" in relation:
        return text_label[0]
    else:
        return f"{text_label[0]} {label_description[relation]} {text_label[1]}"


if __name__ == "__main__":
    clevr_rels = [
        [1, 0, 2, 1, 0, 1, 1, 0, 0, 1, 4],
        [2, 1, 2, 1, 0, 2, 1, 3, 0, 1, 3],
        [2, 1, 7, 0, 0, 1, 0, 5, 0, 1, 4],
    ]

    for clevr_rel in clevr_rels:
        print(clevr_rel_to_text(clevr_rel))
