
pushd ts_app && npm run build
popd

# starts fastapi that serves also the frontend
uvicorn bg.main:fapp --host 0.0.0.0 --port 8080 --workers 1 --app-dir ./backend/ --reload-dir ./backend/bg  --reload --reload-exclude 'test*'

./run_nginx.sh



# alembic  - migrations
alembic init migrations

alembic revision --autogenerate -m "Create a baseline migrations"
alembic upgrade head
