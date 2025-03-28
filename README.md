with openrouter


1->
pyhton -m venv .venv
2->
source .venv/bin/activate
3->
pip install -r requirements.txt
4->
uvicorn fast_api_app:app --reload --host 0.0.0.0 --port 8000