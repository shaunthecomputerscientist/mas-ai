def _is_package_installed(package_name: str) -> bool:
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

# Only expose tools if dependencies are installed
if _is_package_installed('google.oauth2'):
    from .calendarTools import fetch_calendar_events, manage_calendar_event
else:
    def _missing_calendar(*args, **kwargs):
        raise ImportError("Calendar tools require 'google-api-python-client'. Install with: pip install mas-ai[tools]")
    fetch_calendar_events = manage_calendar_event = _missing_calendar

if _is_package_installed('opencv-python'):
    from .visiontools import Vision_Model
else:
    def _missing_vision(*args, **kwargs):
        raise ImportError("Vision tools require 'opencv-python'. Install with: pip install mas-ai[tools]")
    Vision_Model = _missing_vision

# ... similar for other optional tools 