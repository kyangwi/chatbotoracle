import os

flask_tmpl = open('templates/index.html', 'r', encoding='utf-8').read()

django_tmpl = flask_tmpl.replace(
    "var datasetIntroText = {{ (dataset_intro_overview or '')|tojson }};",
    "var datasetIntroText = JSON.parse(document.getElementById('intro-data').textContent);"
).replace(
    "var datasetAnalysisText = {{ (dataset_intro_analysis or '')|tojson }};",
    "var datasetAnalysisText = JSON.parse(document.getElementById('analysis-data').textContent);"
).replace(
    "var datasetSuggestions = {{ (dataset_intro_suggestions or [])|tojson }};",
    "var datasetSuggestions = JSON.parse(document.getElementById('suggestions-data').textContent);"
)

json_scripts = """
    {{ dataset_intro_overview|default:""|json_script:"intro-data" }}
    {{ dataset_intro_analysis|default:""|json_script:"analysis-data" }}
    {{ dataset_intro_suggestions|default:"[]"|json_script:"suggestions-data" }}
"""
django_tmpl = django_tmpl.replace('<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>', json_scripts + '\n    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>')

os.makedirs('chat/templates/chat', exist_ok=True)
open('chat/templates/chat/index.html', 'w', encoding='utf-8').write(django_tmpl)
print('Template successfully migrated')
