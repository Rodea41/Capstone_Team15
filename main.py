import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from dashboard.backend.predictiveDashboard import run_dashboard

if __name__ == "__main__":
    run_dashboard()
