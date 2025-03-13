import os

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {
    'data': os.path.join(BASE_DIR, 'data/data_all.xlsx'),
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
