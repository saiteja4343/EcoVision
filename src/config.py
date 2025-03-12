import os

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {
    'data': os.path.join(BASE_DIR, 'data/data_all.xlsx'),
    'models': os.path.join(BASE_DIR, 'models/'),
    'exports': os.path.join(BASE_DIR, 'exports/')
}

# Theme configurations
THEME = {
    'primary_color': '#2E86AB',
    'secondary_color': '#A23B72',
    'background_color': '#0E1117',
    'secondary_background_color': '#262730',
    'text_color': '#FAFAFA',
    'button_text_color': '#FFFFFF',
    'font_family': 'Helvetica',
    'border_radius': '8px'
}

# # Model default parameters
# MODEL_CONFIG = {
#     'default_confidence': 0.5,
#     'default_imgsz': 640,
#     'frame_skip': 2
# }

# # Camera configurations
# CAMERA_SETTINGS = {
#     'default_resolution': (640, 480),
#     'max_cameras_to_check': 4,
#     'frame_delay': 0.03
# }
#
# # CO2 calculation defaults
# EMISSION_FACTORS = {
#     'default_weight_unit': 'kg',
#     'co2_unit': 'kg CO₂ eq'
# }
