import sys
sys.path.insert(0, '.')
import django, os
os.environ['DJANGO_SETTINGS_MODULE'] = 'sqlchatbot.settings'
django.setup()
from chat.utils import has_visual_intent

tests = [
    ('generate a bargraph for sales by region', True),
    ('generate a bar graph for sales by region', True),
    ('come up with the piechart for that', True),
    ('show me a pie chart of revenue', True),
    ('create a line chart over time', True),
    ('plot a scatter for profit vs cost', True),
    ('what are the top KPIs', False),
    ('how many sales last month', False),
]

all_pass = True
for question, expected in tests:
    result = has_visual_intent(question)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"[{status}] [{('YES' if result else ' NO')}] {question}")

print()
print("All tests passed!" if all_pass else "Some tests FAILED!")
