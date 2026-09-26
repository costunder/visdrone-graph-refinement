VISDRONE_6_CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "car"},
    {"id": 6, "name": "truck"},
    {"id": 9, "name": "bus"},
    {"id": 10, "name": "motor"},
]

VISDRONE_10_CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "people"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "car"},
    {"id": 5, "name": "van"},
    {"id": 6, "name": "truck"},
    {"id": 7, "name": "tricycle"},
    {"id": 8, "name": "awning-tricycle"},
    {"id": 9, "name": "bus"},
    {"id": 10, "name": "motor"},
]

VISDRONE_EVAL_CATEGORIES = VISDRONE_6_CATEGORIES
VISDRONE_EVAL_CATEGORY_IDS = {category["id"] for category in VISDRONE_EVAL_CATEGORIES}

_MODEL_NAME_TO_VISDRONE = {
    "person": {"id": 1, "name": "pedestrian"},
    "pedestrian": {"id": 1, "name": "pedestrian"},
    "bicycle": {"id": 3, "name": "bicycle"},
    "car": {"id": 4, "name": "car"},
    "van": {"id": 5, "name": "van"},
    "truck": {"id": 6, "name": "truck"},
    "tricycle": {"id": 7, "name": "tricycle"},
    "awning-tricycle": {"id": 8, "name": "awning-tricycle"},
    "awningtricycle": {"id": 8, "name": "awning-tricycle"},
    "bus": {"id": 9, "name": "bus"},
    "motorcycle": {"id": 10, "name": "motor"},
    "motor": {"id": 10, "name": "motor"},
}


def _model_class_name(model_names, class_id):
    if isinstance(model_names, dict):
        return model_names.get(class_id, model_names.get(str(class_id), ""))
    if 0 <= class_id < len(model_names):
        return model_names[class_id]
    return ""


def model_class_to_visdrone(model_names, class_id):
    name = str(_model_class_name(model_names, class_id)).strip().lower().replace("_", "-")
    return _MODEL_NAME_TO_VISDRONE.get(name)


def visdrone_category_name(category_id):
    for category in VISDRONE_EVAL_CATEGORIES:
        if category["id"] == category_id:
            return category["name"]
    return "unknown"
