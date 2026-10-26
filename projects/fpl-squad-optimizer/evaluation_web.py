"""Isolated background season-evaluation endpoints for the local Flask app."""
from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import threading
from _thread import LockType
from typing import Any
import uuid
import requests

from flask import Blueprint, jsonify, render_template, request, send_file
from werkzeug.exceptions import BadRequest

from model_registry import MODEL_TYPES
from simulations.evaluate_season import (
    ROOT, capture_inputs, current_context, evaluation_root, readiness,
    snapshot_root, validate_selection, write_report,
)


def create_evaluation_blueprint(
    database: Path, compute_guard: LockType, artifact_lock: AbstractContextManager,
) -> Blueprint:
    blueprint = Blueprint('season_evaluation', __name__)
    state_lock = threading.Lock()
    state: dict[str, Any] = {
        'status': 'idle', 'phase': 'Idle', 'progress': 0,
        'message': 'No season evaluation is running.', 'run_id': None, 'log': [],
    }
    logger = logging.getLogger(__name__)

    @blueprint.errorhandler(BadRequest)
    def malformed_json(error: BadRequest):
        logger.info('Rejected evaluation request: %s', error.name)
        return jsonify(ok=False, error='Malformed JSON request.'), 400

    def snapshot() -> dict:
        with state_lock:
            return {**state, 'log': list(state['log'])}

    def update(**changes) -> None:
        with state_lock:
            state.update(changes)

    def append_log(line: str) -> None:
        clean = line.rstrip()
        with state_lock:
            state['log'] = [*state['log'], clean][-30:]
        if clean.startswith('SEASON_PROGRESS '):
            progress = json.loads(clean[len('SEASON_PROGRESS '):])
            update(progress=int(progress['progress']), phase=progress['phase'])

    def rejection():
        if request.remote_addr not in ('127.0.0.1', '::1', None):
            return jsonify(ok=False, error='Season evaluation actions are localhost-only.'), 403
        origin = request.headers.get('Origin')
        if origin and origin.rstrip('/') != request.host_url.rstrip('/'):
            return jsonify(ok=False, error='Cross-origin evaluation actions are rejected.'), 403
        if not request.is_json:
            return jsonify(ok=False, error='Evaluation actions require JSON.'), 415
        return None

    def run_directory(run_id: str) -> Path:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
            raise ValueError('Invalid evaluation run ID.')
        directory = evaluation_root() / run_id
        if directory.resolve().parent != evaluation_root().resolve():
            raise ValueError('Evaluation run is outside the configured results directory.')
        return directory

    def saved_runs() -> list[dict]:
        root = evaluation_root()
        if not root.exists():
            return []
        paths = sorted(root.glob('*/report.json'), key=lambda path: path.stat().st_mtime, reverse=True)
        rows = []
        for path in paths[:20]:
            run_id = path.parent.name
            try:
                run_directory(run_id)
                report = json.loads(path.read_text(encoding='utf-8'))
                rows.append({
                    'run_id': run_id, 'status': report['status'],
                    'season': report.get('season'), 'models': report.get('models', []),
                    'seeds': report.get('seeds'), 'through': report.get('through'),
                    'created_at': report.get('created_at'),
                })
            except (OSError, ValueError, KeyError) as exc:
                logger.warning('Cannot read evaluation report %s: %s', run_id, exc)
                rows.append({'run_id': run_id, 'status': 'invalid', 'error': str(exc)})
        return rows

    def worker(run_id: str, models: list[str], seeds: int, through: int) -> None:
        directory = run_directory(run_id)
        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            command = [
                sys.executable, '-u', str(ROOT / 'simulations' / 'evaluate_season.py'),
                'evaluate', '--database', str(database), '--snapshots', str(snapshot_root()),
                '--output', str(directory), '--seeds', str(seeds), '--through', str(through),
                '--models', *models,
            ]
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                append_log(line)
            return_code = process.wait()
            path = directory / 'report.json'
            if not path.exists():
                raise RuntimeError(f'Evaluation exited with code {return_code} without a report.')
            report = json.loads(path.read_text(encoding='utf-8'))
            status = report['status']
            if (return_code, status) not in ((0, 'succeeded'), (2, 'blocked'), (1, 'failed')):
                raise RuntimeError(f'Evaluation exited with code {return_code}, status {status}.')
            update(
                status=status, phase=status.capitalize(),
                progress=100 if status == 'succeeded' else 0,
                message=report['message'], completed_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            logger.exception('Season evaluation worker failed')
            append_log(f'ERROR: {exc}')
            update(status='failed', phase='Failed', message=str(exc),
                   completed_at=datetime.now(timezone.utc).isoformat())
            directory.mkdir(parents=True, exist_ok=True)
            write_report(directory / 'report.json', {
                'status': 'failed', 'message': str(exc), 'models': models,
                'seeds': seeds, 'through': through, 'summary': [], 'weekly': [],
            })
        finally:
            compute_guard.release()

    @blueprint.get('/season-evaluation')
    def page():
        return render_template('season_evaluation.html', model_types=MODEL_TYPES)

    @blueprint.get('/api/season-evaluation')
    def context():
        try:
            audit = readiness(database, snapshot_root())
            return jsonify(ok=True, readiness=audit, models=MODEL_TYPES,
                           job=snapshot(), runs=saved_runs())
        except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @blueprint.get('/api/season-evaluation/status')
    def status():
        return jsonify(ok=True, **snapshot())

    @blueprint.post('/api/season-evaluation/start')
    def start():
        rejected = rejection()
        if rejected:
            return rejected
        payload = request.get_json()
        try:
            if not isinstance(payload, dict) or set(payload) - {'models', 'seeds', 'through'}:
                raise ValueError('Provide only models, seeds and through.')
            info = current_context(database)
            models = payload.get('models', ['ensemble'])
            seeds = payload.get('seeds', 10)
            through = payload.get('through', info['latest_completed'])
            validate_selection(models, seeds, through, info['latest_completed'])
        except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
            return jsonify(ok=False, error=str(exc)), 400
        if not compute_guard.acquire(blocking=False):
            return jsonify(ok=False, error='A training or evaluation job is already running.'), 409
        run_id = uuid.uuid4().hex
        update(
            status='running', phase='Checking strict inputs', progress=0, run_id=run_id,
            message='Checking dated input bundles before any training.',
            started_at=datetime.now(timezone.utc).isoformat(), completed_at=None, log=[],
        )
        try:
            threading.Thread(target=worker, args=(run_id, models, seeds, through), daemon=True).start()
        except Exception:
            update(status='failed', phase='Failed', message='The evaluation worker could not start.')
            compute_guard.release()
            raise
        return jsonify(ok=True, **snapshot()), 202

    @blueprint.post('/api/season-evaluation/capture')
    def capture():
        rejected = rejection()
        if rejected:
            return rejected
        if request.get_json() != {}:
            return jsonify(ok=False, error='Capture accepts an empty JSON object only.'), 400
        try:
            with artifact_lock:
                manifest = capture_inputs(database, snapshot_root())
            return jsonify(ok=True, manifest=manifest), 201
        except requests.RequestException as exc:
            return jsonify(ok=False, error=f'Official FPL metadata could not be fetched: {exc}'), 502
        except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @blueprint.get('/api/season-evaluation/runs/<run_id>')
    def report(run_id: str):
        try:
            path = run_directory(run_id) / 'report.json'
            if not path.exists():
                return jsonify(ok=False, error='Evaluation report not found.'), 404
            return jsonify(ok=True, run_id=run_id, report=json.loads(path.read_text(encoding='utf-8')))
        except ValueError as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @blueprint.get('/api/season-evaluation/runs/<run_id>/predictions')
    def predictions(run_id: str):
        try:
            directory = run_directory(run_id)
            path = directory / 'player_predictions.csv'
            report_path = directory / 'report.json'
            if (
                not path.exists() or not report_path.exists()
                or json.loads(report_path.read_text(encoding='utf-8'))['status'] != 'succeeded'
            ):
                return jsonify(ok=False, error='Player predictions are available after a completed run.'), 404
            return send_file(path, as_attachment=True, download_name='player_predictions.csv')
        except ValueError as exc:
            return jsonify(ok=False, error=str(exc)), 400

    return blueprint
