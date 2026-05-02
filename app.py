import os
# Must be set before torch/transformers import their native BLAS/OMP backends
os.environ.setdefault('TOKENIZERS_PARALLELISM',      'false')
os.environ.setdefault('OMP_NUM_THREADS',             '1')
os.environ.setdefault('MKL_NUM_THREADS',             '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS',        '1')
# Allow MPS ops not yet in PyTorch to silently fall back to CPU (required for DistilGPT-2 on Apple Silicon)
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import json
import io
import re
import shutil
import subprocess
import time
import uuid
import zipfile
import base64
import urllib.error
import urllib.request
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, session, stream_with_context

from software.ai.image_trainer        import ImageTrainer
from software.ai.text_trainer         import TextTrainer
from software.ai.finetune_trainer     import FinetuneTrainer
from software.ai.smart_prompt_trainer import SmartPromptTrainer
from software.ai.classifier_trainer  import ClassifierTrainer

_OLLAMA_URL   = os.environ.get('OLLAMA_URL',   'http://localhost:11434')
_OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llava')

app = Flask(
    __name__,
    template_folder='software/ui/templates',
    static_folder='assets',
)
app.secret_key = 'brainy-secret-2024'

_MODELS_DIR = Path('data/models')
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

_sessions: dict[str, dict] = {}


def _sid() -> str:
    if 'id' not in session:
        session['id'] = str(uuid.uuid4())
    return session['id']


def _state() -> dict:
    sid = _sid()
    if sid not in _sessions:
        _sessions[sid] = {'mode': None, 'training_mode': 'scratch', 'input_mode': 'image', 'trainer': None}
    return _sessions[sid]


# ── Model library helpers ───────────────────────────────────────────────────────

def _model_dir(model_id: str) -> Path:
    return _MODELS_DIR / model_id


def _read_meta(model_id: str) -> dict | None:
    p = _model_dir(model_id) / 'metadata.json'
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _write_meta(model_id: str, meta: dict) -> None:
    _model_dir(model_id).mkdir(parents=True, exist_ok=True)
    (_model_dir(model_id) / 'metadata.json').write_text(json.dumps(meta, indent=2))


def _list_models() -> list:
    entries = []
    if _MODELS_DIR.exists():
        for d in _MODELS_DIR.iterdir():
            if d.is_dir():
                meta = _read_meta(d.name)
                if meta:
                    entries.append(meta)
    entries.sort(key=lambda e: e.get('created_at', ''), reverse=True)
    return entries


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get('/')
def index():
    return render_template('trainer.html')


def _make_trainer(mode: str, training_mode: str, input_mode: str = 'image'):
    if mode == 'image':
        return ImageTrainer()
    if mode == 'classifier':
        return ClassifierTrainer(input_mode=input_mode, training_mode=training_mode)
    if training_mode == 'finetune':
        return FinetuneTrainer()
    if training_mode == 'smart_prompt':
        return SmartPromptTrainer()
    return TextTrainer()


@app.post('/api/mode')
def set_mode():
    data          = request.get_json(force=True) or {}
    mode          = data.get('mode', 'image')
    training_mode = data.get('training_mode', 'scratch')
    input_mode    = data.get('input_mode', 'image')
    if mode not in ('image', 'text', 'classifier'):
        return jsonify({'ok': False, 'error': 'Invalid mode.'})
    if training_mode not in ('scratch', 'finetune', 'smart_prompt'):
        training_mode = 'scratch'
    if input_mode not in ('image', 'text', 'audio'):
        input_mode = 'image'
    state = _state()
    if (state['mode'] != mode or state.get('training_mode') != training_mode
            or state.get('input_mode') != input_mode):
        old_trainer    = state.get('trainer')
        old_input_mode = state.get('input_mode')
        state['mode']          = mode
        state['training_mode'] = training_mode
        state['input_mode']    = input_mode
        new_trainer = _make_trainer(mode, training_mode, input_mode)
        # When only the training_mode changes within classifier, preserve labels + examples
        if (mode == 'classifier' and isinstance(old_trainer, ClassifierTrainer)
                and old_input_mode == input_mode):
            new_trainer.labels    = old_trainer.labels[:]
            new_trainer._examples = {k: list(v) for k, v in old_trainer._examples.items()}
        state['trainer'] = new_trainer
    return jsonify({'ok': True, 'mode': mode, 'training_mode': training_mode,
                    'input_mode': input_mode})


@app.post('/api/upload')
def upload():
    data  = request.get_json(force=True) or {}
    state = _state()
    if state['trainer'] is None:
        return jsonify({'ok': False, 'error': 'No mode set.'})
    try:
        if state['mode'] == 'image':
            src = (data.get('image') or '').strip()
            if not src:
                return jsonify({'ok': False, 'error': 'No image data.'})
            state['trainer'].add_image(src)
            idx   = state['trainer'].count() - 1
            thumb = state['trainer'].thumbnail_b64(idx)
            return jsonify({'ok': True, 'count': state['trainer'].count(), 'thumb': thumb})
        else:
            text = (data.get('text') or '').strip()
            if not text:
                return jsonify({'ok': False, 'error': 'No text.'})
            state['trainer'].add_text(text)
            return jsonify({'ok': True, 'count': state['trainer'].count()})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.post('/api/clear')
def clear():
    state = _state()
    if state['trainer']:
        state['trainer'].clear()
    return jsonify({'ok': True})


@app.post('/api/remove')
def remove_data_item():
    data  = request.get_json(force=True) or {}
    state = _state()
    if state['trainer'] is None:
        return jsonify({'ok': False, 'error': 'No mode set.'})
    try:
        idx = int(data.get('idx', -1))
        if state['mode'] == 'image':
            state['trainer'].remove_image(idx)
        elif state['mode'] == 'text':
            state['trainer'].remove_text(idx)
        else:
            return jsonify({'ok': False, 'error': 'Single-item delete is not used for classifier mode.'})
        return jsonify({'ok': True, 'count': state['trainer'].count(), 'trained': state['trainer'].trained})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.get('/api/info')
def info():
    state = _state()
    if not state['trainer']:
        return jsonify({'ok': True, 'mode': None, 'trained': False, 'count': 0})
    return jsonify({'ok': True, 'mode': state['mode'], **state['trainer'].get_info()})


@app.get('/api/train-stream')
def train_stream():
    state = _state()
    if not state['trainer']:
        def _err():
            yield 'event: train_error\ndata: {"message": "No trainer — set a mode first."}\n\n'
        return Response(_err(), mimetype='text/event-stream')

    epochs = int(request.args.get('epochs', 60))
    lr     = float(request.args.get('lr', 0.001))

    if state['mode'] == 'image':
        batch_size     = int(request.args.get('batch_size', 8))
        latent_dim     = int(request.args.get('latent_dim', 128))
        augment_factor = int(request.args.get('augment_factor', 50))
        preview_every  = int(request.args.get('preview_every', 5))
        gen = state['trainer'].train(
            epochs=epochs, lr=lr, batch_size=batch_size,
            latent_dim=latent_dim, augment_factor=augment_factor,
            preview_every=preview_every,
        )
    elif isinstance(state['trainer'], ClassifierTrainer):
        batch_size = int(request.args.get('batch_size', 8))
        gen = state['trainer'].train(epochs=epochs, lr=lr, batch_size=batch_size)
    elif isinstance(state['trainer'], FinetuneTrainer):
        batch_size = int(request.args.get('batch_size', 4))
        max_length = int(request.args.get('max_length', 64))
        gen = state['trainer'].train(
            epochs=epochs, lr=lr,
            batch_size=batch_size, max_length=max_length,
        )
    else:
        hidden_size = int(request.args.get('hidden_size', 40))
        seq_len     = int(request.args.get('seq_len', 50))
        gen = state['trainer'].train(
            epochs=epochs, lr=lr,
            hidden_size=hidden_size, seq_len=seq_len,
        )

    def _stream():
        for event in gen:
            phase = event.get('phase', 'message')
            if phase == 'error':
                phase = 'train_error'
            yield f'event: {phase}\ndata: {json.dumps(event)}\n\n'
        yield 'event: stream_end\ndata: {}\n\n'

    return Response(
        stream_with_context(_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':     'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection':        'keep-alive',
        },
    )


@app.post('/api/generate')
def generate():
    data  = request.get_json(force=True) or {}
    state = _state()
    if not state['trainer']:
        return jsonify({'ok': False, 'error': 'No trainer.'})
    try:
        if state['mode'] == 'image':
            if data.get('type') == 'interpolate':
                steps = int(data.get('steps', 10))
                strip = state['trainer'].interpolate(steps=steps)
                return jsonify({'ok': True, 'strip': strip})
            else:
                n    = int(data.get('n', 16))
                grid = state['trainer'].generate(n=n)
                return jsonify({'ok': True, 'grid': grid})
        else:
            prompt      = data.get('prompt', '')
            length      = int(data.get('length', 200))
            temperature = float(data.get('temperature', 1.0))
            print(f'[/api/generate] trainer type={type(state["trainer"]).__name__}', flush=True)
            text        = state['trainer'].generate(prompt=prompt, length=length, temperature=temperature)
            print(f'[/api/generate] text={text!r}', flush=True)
            result      = {'ok': True, 'text': text}
            # Smart Prompt trainers attach retrieval context to each generation
            extra = getattr(state['trainer'], '_last_generate_meta', None)
            if extra:
                result.update(extra)
            return jsonify(result)
    except RuntimeError as e:
        return jsonify({'ok': False, 'error': str(e)})


# ── Model library ──────────────────────────────────────────────────────────────

_TYPE_MAP  = {'image': 'image_generator', 'text': 'text_generator', 'classifier': 'classifier'}
_MODE_MAP  = {'image_generator': 'image', 'text_generator': 'text', 'classifier': 'classifier'}
_EMOJI_MAP = {
    'image_generator': '🎨',
    'text_generator':  '✍️',
    'audio_generator': '🎵',
    'classifier':      '🏷️',
}


@app.post('/api/save')
def save_model():
    data  = request.get_json(force=True) or {}
    state = _state()
    if not state['trainer'] or not state['trainer'].trained:
        return jsonify({'ok': False, 'error': 'No trained model to save.'})

    name      = (data.get('name') or 'Untitled').strip()[:40]
    model_type = _TYPE_MAP.get(state['mode'], state['mode'])
    emoji      = (data.get('emoji') or '').strip() or _EMOJI_MAP.get(model_type, '🤖')

    model_id      = str(uuid.uuid4())
    mdir          = _model_dir(model_id)
    mdir.mkdir(parents=True, exist_ok=True)
    weights_sub   = getattr(state['trainer'], 'WEIGHTS_SUBPATH', 'weights.pt')
    wpath         = mdir / weights_sub
    training_mode = state.get('training_mode', 'scratch')

    try:
        state['trainer'].save(wpath)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

    info = state['trainer'].get_info()
    meta = {
        'id':            model_id,
        'name':          name,
        'emoji':         emoji,
        'model_type':    model_type,
        'training_mode': training_mode,
        'input_mode':    state.get('input_mode', 'image'),
        'created_at':    time.strftime('%Y-%m-%dT%H:%M:%S'),
        'training_params': {
            'epochs':         data.get('epochs'),
            'lr':             data.get('lr'),
            'batch_size':     data.get('batch_size'),
            'latent_dim':     data.get('latent_dim'),
            'image_size':     128 if state['mode'] == 'image' else None,
            'augment_factor': data.get('augment_factor'),
        },
        'training_stats': {
            'final_loss':       info.get('loss'),
            'final_accuracy':   info.get('final_accuracy'),
            'duration_seconds': data.get('duration_seconds'),
            'dataset_size':     info.get('count'),
            'n_params':         info.get('n_params'),
        },
        'weights_file': weights_sub,
    }
    if state['mode'] == 'classifier':
        meta['labels'] = info.get('labels', [])
    _write_meta(model_id, meta)
    return jsonify({'ok': True, 'entry': meta})


@app.get('/api/smart-status')
def smart_status():
    """Check whether sentence-transformers and Ollama/LLaVA are available."""
    try:
        import sentence_transformers as _st
        st_ok = True
    except ImportError:
        st_ok = False

    ollama_ok = False
    llava_ok  = False
    try:
        with urllib.request.urlopen(f'{_OLLAMA_URL}/api/tags', timeout=3) as resp:
            data        = json.loads(resp.read())
            ollama_ok   = True
            model_names = [m.get('name', '').split(':')[0] for m in data.get('models', [])]
            llava_ok    = _OLLAMA_MODEL.split(':')[0] in model_names
    except Exception:
        pass

    return jsonify({
        'ok':                   True,
        'sentence_transformers': st_ok,
        'ollama':                ollama_ok,
        'llava':                 llava_ok,
        'ollama_url':            _OLLAMA_URL,
        'ollama_model':          _OLLAMA_MODEL,
        'available':             st_ok and ollama_ok and llava_ok,
    })


@app.get('/api/library')
def get_library():
    return jsonify({'ok': True, 'entries': _list_models()})


@app.post('/api/load')
def load_model():
    data     = request.get_json(force=True) or {}
    model_id = data.get('id', '')
    meta     = _read_meta(model_id)
    if not meta:
        return jsonify({'ok': False, 'error': 'Model not found.'})

    weights_sub = meta.get('weights_file', 'weights.pt')
    wpath = _model_dir(model_id) / weights_sub
    if not wpath.exists():
        return jsonify({'ok': False, 'error': 'Model weights file missing from disk.'})

    mode          = _MODE_MAP.get(meta.get('model_type', ''), 'image')
    training_mode = meta.get('training_mode', 'scratch')
    input_mode    = meta.get('input_mode', 'image')
    s = _state()
    s['mode']          = mode
    s['training_mode'] = training_mode
    s['input_mode']    = input_mode
    s['trainer']       = _make_trainer(mode, training_mode, input_mode)

    try:
        s['trainer'].load(wpath)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

    return jsonify({'ok': True, 'mode': mode, 'training_mode': training_mode,
                    'input_mode': input_mode, 'entry': meta})


@app.post('/api/rename')
def rename_model():
    data     = request.get_json(force=True) or {}
    model_id = data.get('id', '')
    meta     = _read_meta(model_id)
    if not meta:
        return jsonify({'ok': False, 'error': 'Model not found.'})

    new_name  = (data.get('name')  or '').strip()[:40]
    new_emoji = (data.get('emoji') or '').strip()
    if new_name:
        meta['name'] = new_name
    if new_emoji:
        meta['emoji'] = new_emoji
    _write_meta(model_id, meta)
    return jsonify({'ok': True, 'entry': meta})


@app.post('/api/delete')
def delete_model():
    data     = request.get_json(force=True) or {}
    model_id = data.get('id', '')
    mdir     = _model_dir(model_id)
    if not mdir.exists():
        return jsonify({'ok': False, 'error': 'Not found.'})
    shutil.rmtree(mdir)
    return jsonify({'ok': True})


# ── Classifier routes ──────────────────────────────────────────────────────────

@app.post('/api/classifier/labels')
def classifier_labels():
    data  = request.get_json(force=True) or {}
    state = _state()
    if not isinstance(state.get('trainer'), ClassifierTrainer):
        return jsonify({'ok': False, 'error': 'No classifier active.'})
    labels = data.get('labels', [])
    if not labels or not isinstance(labels, list):
        return jsonify({'ok': False, 'error': 'labels must be a non-empty list.'})
    state['trainer'].set_labels([str(l).strip() for l in labels if str(l).strip()])
    return jsonify({'ok': True, 'labels': state['trainer']._labels})


@app.post('/api/classifier/example')
def classifier_example():
    data  = request.get_json(force=True) or {}
    state = _state()
    if not isinstance(state.get('trainer'), ClassifierTrainer):
        return jsonify({'ok': False, 'error': 'No classifier active.'})
    label = (data.get('label') or '').strip()
    raw   = (data.get('data')  or '').strip()
    if not label or not raw:
        return jsonify({'ok': False, 'error': 'label and data are required.'})
    action = data.get('action', 'add')
    if action == 'remove':
        idx = int(data.get('idx', 0))
        state['trainer'].remove_example(label, idx)
        return jsonify({'ok': True, 'counts': state['trainer'].count_per_label()})
    try:
        state['trainer'].add_example(label, raw)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})
    return jsonify({'ok': True, 'counts': state['trainer'].count_per_label()})


@app.post('/api/classifier/rename-label')
def rename_label():
    data  = request.get_json(force=True) or {}
    state = _state()
    if not isinstance(state.get('trainer'), ClassifierTrainer):
        return jsonify({'ok': False, 'error': 'No classifier active.'})
    old_label = (data.get('old_label') or '').strip()
    new_label = (data.get('new_label') or '').strip()
    if not old_label or not new_label:
        return jsonify({'ok': False, 'error': 'old_label and new_label required.'})
    trainer = state['trainer']
    if old_label not in trainer.labels:
        return jsonify({'ok': False, 'error': f'Label "{old_label}" not found.'})
    if new_label in trainer.labels:
        return jsonify({'ok': False, 'error': f'Label "{new_label}" already exists.'})
    idx = trainer.labels.index(old_label)
    trainer.labels[idx] = new_label
    trainer._examples[new_label] = trainer._examples.pop(old_label, [])
    return jsonify({'ok': True, 'labels': trainer.labels})


@app.post('/api/predict')
def predict():
    data  = request.get_json(force=True) or {}
    state = _state()
    if not isinstance(state.get('trainer'), ClassifierTrainer):
        return jsonify({'ok': False, 'error': 'No classifier active.'})
    raw = (data.get('data') or '').strip()
    if not raw:
        return jsonify({'ok': False, 'error': 'data is required.'})
    try:
        result = state['trainer'].predict(raw)
        return jsonify({'ok': True, **result})
    except RuntimeError as e:
        return jsonify({'ok': False, 'error': str(e)})


# ── Bot designs ────────────────────────────────────────────────────────────────

_BOTS_DIR = Path('data/bots')
_BOTS_DIR.mkdir(parents=True, exist_ok=True)


@app.get('/api/bots')
def list_bots():
    entries = []
    if _BOTS_DIR.exists():
        for f in sorted(_BOTS_DIR.glob('*.json'),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                entries.append(json.loads(f.read_text()))
            except Exception:
                pass
    return jsonify({'ok': True, 'bots': entries})


@app.post('/api/bots/save')
def save_bot():
    data   = request.get_json(force=True) or {}
    bot_id = data.get('id') or str(uuid.uuid4())
    _BOTS_DIR.mkdir(parents=True, exist_ok=True)
    p   = _BOTS_DIR / f'{bot_id}.json'
    bot = {**data, 'id': bot_id, 'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    p.write_text(json.dumps(bot, indent=2))
    return jsonify({'ok': True, 'bot': bot})


@app.post('/api/bots/delete')
def delete_bot():
    data   = request.get_json(force=True) or {}
    bot_id = data.get('id', '')
    p      = _BOTS_DIR / f'{bot_id}.json'
    if p.exists():
        p.unlink()
    return jsonify({'ok': True})


# ── Pi export helpers ──────────────────────────────────────────────────────────

_PI_RUN_PY = r'''#!/usr/bin/env python3
"""Brainy Bot Runtime — auto-generated by Brainy.  Run: python3 run.py"""
import json, math, os, re, subprocess, sys, tempfile, time, urllib.request, urllib.error
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
BOT_DIR       = Path(__file__).parent
bot           = json.loads((BOT_DIR / 'bot.json').read_text())
brain         = bot.get('brain') or {}
MODEL_TYPE    = brain.get('model_type',    'text_generator')
TRAINING_MODE = brain.get('training_mode', 'smart_prompt')
WEIGHTS_FILE  = brain.get('weights_file',  'weights.pt')
BOT_NAME      = bot.get('name', 'My Bot')
OLLAMA_URL    = os.environ.get('OLLAMA_URL',   'http://localhost:11434')
OLLAMA_MODEL  = os.environ.get('OLLAMA_MODEL', 'llava')

_inputs  = {i['id']: i for i in bot.get('inputs',  [])}
_outputs = {o['id']: o for o in bot.get('outputs', [])}
_rules   = bot.get('rules', [])

print(f'Brainy: {BOT_NAME}  [{MODEL_TYPE}/{TRAINING_MODE}]')
print(f'Inputs:  {[v["type"] for v in _inputs.values()]}')
print(f'Outputs: {[v["type"] for v in _outputs.values()]}')
print()

# ── Audio helpers ──────────────────────────────────────────────────────────────
def _usb_card():
    """Return card number of first USB audio device, or None."""
    try:
        out = subprocess.check_output(['arecord', '-l'], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if 'usb' in line.lower():
                m = re.search(r'card (\d+)', line)
                if m:
                    return int(m.group(1))
    except Exception:
        pass
    return None

_CARD = _usb_card()
if _CARD is not None:
    print(f'Audio: USB card {_CARD}')

def speak(text):
    """Speak via Piper neural TTS (offline). Falls back to espeak."""
    print(f'[speaker] {text}')
    env = dict(os.environ)
    if _CARD is not None:
        env['AUDIODEV'] = f'hw:{_CARD},0'

    piper_model = os.path.expanduser(
        os.environ.get('PIPER_MODEL', '~/piper-voices/en_US-amy-medium.onnx')
    )
    aplay_device = f'plughw:{_CARD},0' if _CARD is not None else 'default'

    # Try Piper first (neural, natural voice)
    if Path(piper_model).exists():
        try:
            piper_path = os.path.expanduser('~/.local/bin/piper')
            if not Path(piper_path).exists():
                piper_path = 'piper'  # fallback to PATH lookup

            piper_proc = subprocess.Popen(
                [piper_path, '--model', piper_model, '--output_raw'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            aplay_proc = subprocess.Popen(
                ['aplay', '-D', aplay_device, '-r', '22050',
                 '-f', 'S16_LE', '-t', 'raw', '-'],
                stdin=piper_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            piper_proc.stdin.write(text.encode())
            piper_proc.stdin.close()
            aplay_proc.wait(timeout=60)
            return
        except Exception as e:
            print(f'[piper error: {e}, falling back to espeak]')

    # Fallback to espeak
    try:
        subprocess.run(['espeak', '-s', '150', '--', text],
                       env=env, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

_vosk_model = None
_vosk_failed = False

def _get_vosk_model():
    global _vosk_model, _vosk_failed
    if _vosk_failed:
        return None
    if _vosk_model is not None:
        return _vosk_model
    try:
        import vosk
        model_path = os.environ.get('VOSK_MODEL', str(Path.home() / 'vosk-models' / 'vosk-model-small-en-us-0.15'))
        if not Path(model_path).exists():
            _vosk_failed = True
            return None
        vosk.SetLogLevel(-1)
        _vosk_model = vosk.Model(model_path)
        return _vosk_model
    except Exception as exc:
        print('[vosk init error: ' + str(exc) + ']')
        _vosk_failed = True
        return None


def _transcribe_vosk(wav_path):
    """Transcribe a WAV file with Vosk. Returns text or None on failure."""
    import wave, json as _json
    model = _get_vosk_model()
    if model is None:
        return None
    try:
        import vosk
        wf = wave.open(wav_path, 'rb')
        rec = vosk.KaldiRecognizer(model, wf.getframerate())
        result_text = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                r = _json.loads(rec.Result())
                if r.get('text'):
                    result_text.append(r['text'])
        final = _json.loads(rec.FinalResult())
        if final.get('text'):
            result_text.append(final['text'])
        wf.close()
        return ' '.join(result_text).strip()
    except Exception as exc:
        print('[vosk transcribe error: ' + str(exc) + ']')
        return None


def listen_mic(secs=5):
    """Record from mic (arecord) and transcribe with Vosk (offline) / Google fallback."""
    device = f'plughw:{_CARD},0' if _CARD is not None else 'default'
    wav = tempfile.mktemp(suffix='.wav')
    print(f'[mic] Recording {secs}s from {device}...')
    try:
        subprocess.run(
            ['arecord', '-D', device, '-f', 'S16_LE', '-r', '16000',
             '-c', '1', '-d', str(secs), wav],
            check=True, capture_output=True)
        # Try Vosk first (offline). Fall back to Google if Vosk unavailable.
        text = _transcribe_vosk(wav)
        if text is None:
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                with sr.AudioFile(wav) as src:
                    audio = r.record(src)
                text = r.recognize_google(audio)
            except Exception as exc:
                print('[mic] Error: ' + str(exc))
                return None
        print(f'[mic] Heard: {text}')
        return text
    except Exception as e:
        print(f'[mic] Error: {e}')
        return None
    finally:
        Path(wav).unlink(missing_ok=True)

def capture_image():
    """Capture from Pi Camera (rpicam-still) or USB webcam (cv2). Returns PIL Image."""
    try:
        from PIL import Image as _PIL
        tmp = tempfile.mktemp(suffix='.jpg')
        try:
            subprocess.run(
                ['rpicam-still', '-o', tmp, '--width', '224',
                 '--height', '224', '-t', '500', '--nopreview'],
                check=True, capture_output=True)
            img = _PIL.open(tmp).convert('RGB').resize((224, 224))
            Path(tmp).unlink(missing_ok=True)
            return img
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
            return _PIL.fromarray(rgb)
    except Exception as e:
        print(f'[camera] {e}')
    return None

def wait_button(pin=17):
    """Block until GPIO button on the given pin is pressed. Falls back to Enter."""
    try:
        from gpiozero import Button as _Btn
        print(f'[button] Waiting for GPIO {pin}...')
        _Btn(pin).wait_for_press()
    except Exception:
        input('[button] Press Enter...')

# ── Model runners ──────────────────────────────────────────────────────────────

class _SmartPromptRAG:
    """RAG: sentence-transformer embeddings + Ollama generation."""
    def __init__(self):
        self._texts = []
        self._embs  = []
        self._mdl   = None

    def load(self, path):
        d = json.loads(Path(path).read_text())
        self._texts = d['texts']
        self._embs  = d['embeddings']
        print('Loading embedding model (all-MiniLM-L6-v2)...')
        from sentence_transformers import SentenceTransformer
        self._mdl = SentenceTransformer('all-MiniLM-L6-v2')
        print(f'Ready — {len(self._texts)} examples.')

    @staticmethod
    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)) + 1e-9)

    def generate(self, prompt='', **_):
        q   = self._mdl.encode(prompt or 'write a new example', normalize_embeddings=True).tolist()
        top = sorted(range(len(self._embs)), key=lambda i: -self._cos(q, self._embs[i]))[:20]
        examples = chr(10).join(self._texts[i] for i in top)
        TRIGGER_WORDS = {'hi', 'hello', 'hey', 'fortuna', 'malafortuna', 'alakazam', 'test', 'go', 'start'}
        clean_prompt = (prompt or '').strip().lower()
        is_trigger = clean_prompt in TRIGGER_WORDS or len(clean_prompt.split()) < 2
        system = (
            'Continue the list. Write ONE more line in exactly the same style, length, and tone as these:'
            + chr(10) + chr(10)
            + examples
            + chr(10)
        )
        if not is_trigger and prompt:
            system += chr(10) + 'Loose theme to consider: ' + prompt + chr(10)
        system += chr(10) + 'Next line:'
        body = json.dumps({
            'model': OLLAMA_MODEL, 'prompt': system, 'stream': False,
            'options': {'temperature': 0.85, 'top_p': 0.9, 'num_predict': 60,
                        'stop': [chr(10) + chr(10), 'Next line:', 'Example']},
        }).encode()
        req = urllib.request.Request(
            OLLAMA_URL + '/api/generate', data=body,
            headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                out = json.loads(r.read()).get('response', '').strip()
                # Detect refusal and retry once with stronger framing
                REFUSAL_PHRASES = (
                    "i can't", "i cannot", "i can not", "i'm not able",
                    "i am not able", "sorry", "i won't", "i will not",
                    "i don't feel comfortable", "i'm unable", "i am unable",
                    "as an ai", "i'm just", "i am just",
                )
                if any(out.lower().lstrip().startswith(p) for p in REFUSAL_PHRASES) or len(out.strip()) < 8:
                    retry_system = system + chr(10) + 'You will'
                    retry_body = json.dumps({
                        'model': OLLAMA_MODEL, 'prompt': retry_system, 'stream': False,
                        'options': {'temperature': 0.7, 'top_p': 0.9, 'num_predict': 60, 'stop': [chr(10) + chr(10), 'Next line:', 'Example']},
                    }).encode()
                    retry_req = urllib.request.Request(
                        OLLAMA_URL + '/api/generate', data=retry_body,
                        headers={'Content-Type': 'application/json'}, method='POST')
                    try:
                        with urllib.request.urlopen(retry_req, timeout=90) as r2:
                            retry_out = json.loads(r2.read()).get('response', '').strip()
                            retry_is_refusal = (
                                any(retry_out.lower().lstrip().startswith(p) for p in REFUSAL_PHRASES)
                                or len(retry_out.strip()) < 8
                            )
                            if retry_is_refusal:
                                import random
                                out = random.choice(self._texts) if self._texts else 'You will encounter unexpected silence.'
                            else:
                                if retry_out and not retry_out.lower().startswith('you will'):
                                    retry_out = 'You will ' + retry_out.lstrip()
                                out = retry_out
                    except Exception:
                        pass
                if chr(10) in out:
                    out = out.split(chr(10))[0].strip()
                while out and out[0] in '-*' + chr(8226) + '0123456789. )':
                    out = out[1:].strip()
                if len(out) > 1 and out[0] in ('"', "'") and out[-1] == out[0]:
                    out = out[1:-1].strip()
                for prefix in ('Here is', "Here's", 'Sure', 'Of course', 'Output:', 'Next line', 'New line'):
                    if out.lower().startswith(prefix.lower()):
                        if ':' in out:
                            out = out.split(':', 1)[1].strip()
                        else:
                            out = out[len(prefix):].strip()
                return out
        except urllib.error.URLError:
            raise RuntimeError('Cannot reach Ollama at ' + OLLAMA_URL + '. Run: ollama serve')


class _LSTMRunner:
    """Scratch text generator: compact token-level LSTM + GPT-2 tokenizer vocab."""
    def __init__(self):
        self._mdl = None
        self._tok = None
        self._t2l = {}
        self._l2t = {}
        self._eos = 0

    def load(self, path):
        import torch
        import torch.nn as nn
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self._t2l = ck['tok2loc']
        self._l2t = {int(k): v for k, v in ck['loc2tok'].items()}
        self._eos = ck['eos_loc']
        vs = len(self._t2l)

        class _Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.embed = nn.Embedding(vs, 64)
                s.lstm  = nn.LSTM(64, 256, 2, batch_first=True, dropout=0.1)
                s.head  = nn.Linear(256, vs)
            def forward(s, x, hidden=None):
                out, hidden = s.lstm(s.embed(x), hidden)
                return s.head(out), hidden

        self._mdl = _Net()
        self._mdl.load_state_dict(ck['model_state'])
        self._mdl.eval()
        from transformers import GPT2Tokenizer
        self._tok = GPT2Tokenizer.from_pretrained('distilgpt2', use_fast=False)
        self._tok.pad_token = self._tok.eos_token
        print('LSTM ready.')

    def generate(self, prompt='', length=200, temperature=1.0, **_):
        import torch
        if prompt:
            gpt_ids = self._tok.encode(prompt)
            lids    = [self._t2l.get(t, self._eos) for t in gpt_ids] or [self._eos]
        else:
            non_eos = [i for i in range(len(self._t2l)) if i != self._eos]
            lids    = non_eos[:1] or [0]
        x      = torch.tensor([lids])
        hidden = None
        gen    = list(lids)
        with torch.no_grad():
            if len(lids) > 1:
                _, hidden = self._mdl(x[:, :-1], hidden)
                x = x[:, -1:]
            for _ in range(length):
                logits, hidden = self._mdl(x, hidden)
                logits = logits[0, -1] / max(temperature, 1e-6)
                nxt = torch.multinomial(torch.softmax(logits, -1), 1).item()
                if nxt == self._eos:
                    break
                gen.append(nxt)
                x = torch.tensor([[nxt]])
        gpt_ids = [self._l2t[i] for i in gen if i in self._l2t]
        return self._tok.decode(gpt_ids, skip_special_tokens=True).strip()


class _FinetuneGPT2Runner:
    """Fine-tuned DistilGPT-2 text generator."""
    def __init__(self):
        self._mdl = None
        self._tok = None

    def load(self, path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(str(path))
        self._mdl = AutoModelForCausalLM.from_pretrained(str(path))
        self._mdl.eval()
        print('DistilGPT-2 ready.')

    def generate(self, prompt='', length=100, temperature=0.9, **_):
        import torch
        if prompt:
            ids = self._tok.encode(prompt, return_tensors='pt')
        else:
            ids = torch.tensor([[self._tok.eos_token_id]])
        prompt_len = ids.shape[1]
        with torch.no_grad():
            out = self._mdl.generate(
                ids, max_new_tokens=length, do_sample=True,
                temperature=max(temperature, 1e-6), top_p=0.9,
                pad_token_id=self._tok.eos_token_id)
        new_tokens = out[0][prompt_len:]
        return self._tok.decode(new_tokens, skip_special_tokens=True).strip()


class _VAERunner:
    """VAE image generator (128x128 px)."""
    def __init__(self):
        self._mdl = None
        self.ld   = 128

    def load(self, path):
        import torch
        import torch.nn as nn
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.ld = ck['latent_dim']
        ld = self.ld

        class _VAE(nn.Module):
            def __init__(s):
                super().__init__()
                s.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 4, 2, 1),    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, 2, 1),   nn.ReLU(),
                    nn.Conv2d(64, 128, 4, 2, 1),  nn.ReLU(),
                    nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
                    nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(),
                    nn.Flatten())
                s.fc_mu     = nn.Linear(512 * 4 * 4, ld)
                s.fc_logvar = nn.Linear(512 * 4 * 4, ld)
                s.fc_decode = nn.Linear(ld, 512 * 4 * 4)
                s.decoder = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.ReLU(),
                    nn.ConvTranspose2d(32, 3, 4, 2, 1),    nn.Sigmoid())
            def decode(s, z):
                return s.decoder(s.fc_decode(z).view(-1, 512, 4, 4))

        self._mdl = _VAE()
        self._mdl.load_state_dict(ck['model_state'])
        self._mdl.eval()
        print('VAE ready.')

    def generate(self, **_):
        import torch
        import numpy as np
        from PIL import Image as _PIL
        with torch.no_grad():
            z   = torch.randn(1, self.ld)
            img = self._mdl.decode(z)[0].permute(1, 2, 0).numpy()
        return _PIL.fromarray((img * 255).astype(np.uint8))


class _ClassifierRunner:
    """Image / text classifier (scratch or fine-tune mode)."""
    def __init__(self):
        self._mdl   = None
        self.labels = []
        self._vocab = {}
        self.imode  = 'image'
        self.tmode  = 'scratch'

    def load(self, path):
        import torch
        import torch.nn as nn
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.labels = ck['labels']
        self.imode  = ck.get('input_mode',    'image')
        self.tmode  = ck.get('training_mode', 'scratch')
        self._vocab = ck.get('vocab', {})
        nc = ck['n_classes']

        if self.imode in ('image', 'audio'):
            if self.tmode == 'finetune':
                from torchvision import models
                bb = models.mobilenet_v3_small(weights=None)
                bb.classifier[3] = nn.Linear(bb.classifier[3].in_features, nc)
                self._mdl = bb
            else:
                class _TinyCNN(nn.Module):
                    def __init__(s):
                        super().__init__()
                        s.features = nn.Sequential(
                            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(4))
                        s.head = nn.Sequential(
                            nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU(),
                            nn.Dropout(0.3), nn.Linear(128, nc))
                    def forward(s, x): return s.head(s.features(x))
                self._mdl = _TinyCNN()
        elif self.imode == 'text':
            if self.tmode == 'finetune':
                class _BertMLP(nn.Module):
                    def __init__(s):
                        super().__init__()
                        s.net = nn.Sequential(
                            nn.Linear(768, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, nc))
                    def forward(s, x): return s.net(x)
                self._mdl = _BertMLP()
            else:
                vs = max(len(self._vocab), 2)
                class _TextMLP(nn.Module):
                    def __init__(s):
                        super().__init__()
                        s.emb  = nn.Embedding(vs, 64, padding_idx=0)
                        s.head = nn.Sequential(
                            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, nc))
                    def forward(s, x):
                        if x.dim() == 1:
                            x = x.unsqueeze(0)
                        mask = (x != 0).float().unsqueeze(-1)
                        emb  = (s.emb(x) * mask).sum(1) / mask.sum(1).clamp(min=1)
                        return s.head(emb)
                self._mdl = _TextMLP()

        self._mdl.load_state_dict(ck['model_state'])
        self._mdl.eval()
        print(f'Classifier ready — labels: {self.labels}')

    def predict_image(self, img):
        import torch
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)])
        with torch.no_grad():
            out = self._mdl(tf(img).unsqueeze(0))
        return self.labels[out[0].argmax().item()]

    def predict_text(self, text):
        import torch
        if self.tmode == 'finetune':
            from transformers import DistilBertTokenizer, DistilBertModel
            tok = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            mdl = DistilBertModel.from_pretrained('distilbert-base-uncased')
            enc = tok(text, return_tensors='pt', truncation=True,
                      max_length=64, padding='max_length')
            with torch.no_grad():
                emb    = mdl(**enc).last_hidden_state[:, 0]
                logits = self._mdl(emb)
        else:
            ws = [self._vocab.get(w, 1) for w in text.lower().split() or ['<unk>']]
            t  = torch.zeros(1, max(len(ws), 1), dtype=torch.long)
            for i, w in enumerate(ws[:t.shape[1]]):
                t[0, i] = w
            with torch.no_grad():
                logits = self._mdl(t)
        return self.labels[logits[0].argmax().item()]


# ── Load brain ─────────────────────────────────────────────────────────────────
print('Loading brain...')
_brain = None
try:
    wp = BOT_DIR / WEIGHTS_FILE
    if MODEL_TYPE == 'text_generator':
        if TRAINING_MODE == 'smart_prompt':
            _brain = _SmartPromptRAG(); _brain.load(wp)
        elif TRAINING_MODE == 'finetune':
            _brain = _FinetuneGPT2Runner(); _brain.load(wp)
        else:
            _brain = _LSTMRunner(); _brain.load(wp)
    elif MODEL_TYPE == 'image_generator':
        _brain = _VAERunner(); _brain.load(wp)
    elif MODEL_TYPE == 'classifier':
        _brain = _ClassifierRunner(); _brain.load(wp)
    else:
        print(f'Warning: unknown model_type "{MODEL_TYPE}"')
except Exception as e:
    print(f'ERROR loading brain: {e}', file=sys.stderr)
    sys.exit(1)

print('Brain ready.\n')

# ── Output dispatch ────────────────────────────────────────────────────────────
_printer_instance = None
_printer_failed = False

def _get_printer():
    """Return a single persistent Usb printer, or None if unavailable.
    Reusing one instance avoids 'Resource busy' on the second print."""
    global _printer_instance, _printer_failed
    if _printer_failed:
        return None
    if _printer_instance is not None:
        return _printer_instance
    try:
        from escpos.printer import Usb
        vid = int(os.environ.get('PRINTER_VID', '0x0485'), 16)
        pid = int(os.environ.get('PRINTER_PID', '0x5741'), 16)
        profile = os.environ.get('PRINTER_PROFILE', 'TM-T88III')
        _printer_instance = Usb(vid, pid, profile=profile)
        return _printer_instance
    except Exception as exc:
        print('[printer init error: ' + str(exc) + ']')
        _printer_failed = True
        return None


def _send_output(out_type, content):
    if out_type == 'speaker':
        speak(str(content))
        return
    if out_type == 'printer':
        text = str(content)
        printer = _get_printer()
        if printer is None:
            print('[printer fallback] ' + text)
            return
        try:
            logo_candidates = [
                BOT_DIR / 'logobrainy.png',
                BOT_DIR.parent / 'logobrainy.png',
                Path(__file__).resolve().parent / 'logobrainy.png',
                Path.home() / 'logobrainy.png',
            ]
            logo_path = next((p for p in logo_candidates if p.exists()), None)
            if logo_path is not None:
                try:
                    printer.set(align='center', bold=False, double_height=False, double_width=False)
                    printer.image(str(logo_path))
                except Exception as img_exc:
                    print('[logo print error: ' + str(img_exc) + ']')
            printer.set(align='center', bold=True, double_height=True, double_width=True)
            printer.text(BOT_NAME.upper() + chr(10))
            printer.set(align='center', font='b', bold=False, double_height=False, double_width=False)
            printer.text(time.strftime('%I:%M %p').lstrip('0') + chr(10))
            printer.text('-' * 32 + chr(10))
            # Body (smaller font)
            printer.set(align='left', font='b', bold=False, double_height=False, double_width=False)
            printer.text(text + chr(10))
            printer.set(align='center', font='b', bold=False, double_height=False, double_width=False)
            printer.text('-' * 32 + chr(10))
            printer.text(chr(10) * 3)
            printer.cut()
        except Exception as exc:
            print('[printer error: ' + str(exc) + ']')
            print('[printer fallback] ' + text)
            globals()['_printer_instance'] = None
        return
    if out_type == 'screen':
        if hasattr(content, 'save'):
            p = BOT_DIR / 'output.png'
            content.save(p)
            print('[screen] Image saved to ' + str(p))
            for viewer in (['eog', str(p)], ['display', str(p)], ['feh', str(p)]):
                try:
                    subprocess.Popen(viewer)
                    break
                except FileNotFoundError:
                    pass
        else:
            print(chr(10) + '--- SCREEN ---' + chr(10) + str(content) + chr(10))
        return
    print('[' + out_type + '] ' + str(content))


def _fire_rules(inp_id, data):
    """Run all rules attached to inp_id and dispatch outputs."""
    cache = {}  # generate once per (action, content) — all outputs share the same result
    for rule in _rules:
        if rule.get('inputId') != inp_id:
            continue
        out_cfg   = _outputs.get(rule.get('outputId', ''), {})
        out_type  = out_cfg.get('type', 'printer')
        action    = rule.get('action', 'model_out')
        cache_key = (action, rule.get('content', ''))

        if cache_key in cache:
            result = cache[cache_key]
        elif action == 'model_out' and _brain is not None:
            try:
                if MODEL_TYPE == 'text_generator':
                    result = _brain.generate(prompt=str(data) if data else '')
                elif MODEL_TYPE == 'image_generator':
                    result = _brain.generate()
                elif MODEL_TYPE == 'classifier':
                    if isinstance(data, str):
                        result = _brain.predict_text(data)
                    else:
                        result = _brain.predict_image(data)
                else:
                    result = '[unsupported model type]'
            except Exception as exc:
                result = f'[error: {exc}]'
                print(f'[inference error] {exc}', file=sys.stderr)
            cache[cache_key] = result
        else:
            result = rule.get('content') or str(data or '')
            cache[cache_key] = result

        print(f'[rule] {inp_id[:8]} -> {out_type}: {str(result)[:80]}')
        _send_output(out_type, result)


# ── Event loop ─────────────────────────────────────────────────────────────────
print('Running — Ctrl+C to stop.\n')
try:
    while True:
        for inp_id, inp in _inputs.items():
            t = inp.get('type', 'textinput')

            if t == 'textinput':
                try:
                    text = input(f'[{BOT_NAME}] > ').strip()
                except EOFError:
                    sys.exit(0)
                if text:
                    _fire_rules(inp_id, text)

            elif t == 'microphone':
                data = listen_mic()
                if data:
                    keyword = (inp.get('keyword') or '').strip().lower()
                    if keyword:
                        text_clean = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in str(data).lower())
                        if keyword not in text_clean.split() and keyword not in text_clean:
                            time.sleep(0.3)
                            continue
                    _fire_rules(inp_id, data)
                time.sleep(0.3)

            elif t == 'button':
                wait_button()
                _fire_rules(inp_id, None)

            elif t == 'camera':
                img = capture_image()
                if img is not None:
                    _fire_rules(inp_id, img)
                else:
                    print('[camera] No image captured — retrying in 1s')
                time.sleep(1.0)

            else:
                print(f'[{t}] unknown input type — sleeping 2s')
                time.sleep(2.0)

except KeyboardInterrupt:
    print('\nBot stopped.')
'''


def _pi_collect_weights(bot: dict) -> list:
    """Return [(zip_path, local_Path)] for all model files this brain needs.
    Raises ValueError with a user-friendly message if weights are missing."""
    brain         = bot.get('brain') or {}
    model_id      = brain.get('id', '')
    weights_file  = brain.get('weights_file', '')
    training_mode = brain.get('training_mode', 'scratch')
    model_type    = brain.get('model_type', 'text_generator')

    if not model_id:
        raise ValueError('Cannot export — no brain attached to this bot.')

    mdir = _MODELS_DIR / model_id
    if not mdir.exists():
        raise ValueError(
            'Cannot export — brain not found in the Library. '
            'Make sure the brain is saved before exporting.'
        )
    if not weights_file:
        raise ValueError('Cannot export — brain metadata is missing weights_file.')

    # Fine-tune text_generator saves an HF directory; everything else is a single file
    if training_mode == 'finetune' and model_type == 'text_generator':
        hf_dir = mdir / weights_file
        if not hf_dir.is_dir():
            raise ValueError(
                f'Cannot export — fine-tuned weights directory "{weights_file}" not found. '
                'Re-train and save the brain first.'
            )
        files = [
            (f'{weights_file}/{f.name}', f)
            for f in sorted(hf_dir.iterdir()) if f.is_file()
        ]
        if not files:
            raise ValueError('Cannot export — fine-tuned weights directory is empty.')
        return files

    # Single-file weights (.pt or .json)
    wpath = mdir / weights_file
    if not wpath.exists():
        raise ValueError(
            "Cannot export — this brain hasn't been trained yet. "
            'Train the brain and click Save before exporting.'
        )
    return [(weights_file, wpath)]


def _pi_requirements(bot: dict) -> str:
    """Generate requirements.txt based on the bot's model type and inputs/outputs."""
    brain         = bot.get('brain') or {}
    model_type    = brain.get('model_type',    'text_generator')
    training_mode = brain.get('training_mode', 'smart_prompt')
    input_mode    = brain.get('input_mode',    'text')
    inp_types     = {i.get('type') for i in bot.get('inputs',  [])}
    out_types     = {o.get('type') for o in bot.get('outputs', [])}

    pkgs: set[str] = set()

    if training_mode == 'smart_prompt':
        pkgs.add('sentence-transformers')   # brings torch as a dependency
    elif model_type == 'text_generator':    # scratch LSTM or finetune GPT-2
        pkgs.update(['torch', 'transformers'])
    elif model_type == 'image_generator':
        pkgs.update(['torch', 'numpy', 'Pillow'])
    elif model_type == 'classifier':
        pkgs.update(['torch', 'Pillow'])
        if training_mode == 'finetune' and input_mode in ('image', 'audio'):
            pkgs.add('torchvision')
        if training_mode == 'finetune' and input_mode == 'text':
            pkgs.add('transformers')
        if training_mode != 'finetune' and input_mode in ('image', 'audio'):
            pkgs.add('torchvision')   # for transforms.Resize / ToTensor

    if 'microphone' in inp_types:
        pkgs.update(['SpeechRecognition', 'PyAudio', 'vosk'])
    if 'camera' in inp_types:
        pkgs.update(['opencv-python', 'Pillow'])
    if model_type == 'image_generator' and 'screen' in out_types:
        pkgs.add('Pillow')
    if 'printer' in out_types:
        pkgs.add('python-escpos')
    if 'speaker' in out_types:
        pkgs.add('piper-tts')

    return '\n'.join(sorted(pkgs)) + '\n'


def _pi_readme(bot: dict, weight_names: list) -> str:
    """Generate a complete, accurate README.txt for the Pi package."""
    name    = bot.get('name', 'My Bot')
    brain   = bot.get('brain') or {}
    m_type  = brain.get('model_type',    'text_generator')
    t_mode  = brain.get('training_mode', 'smart_prompt')
    inputs  = [i.get('type', '?') for i in bot.get('inputs',  [])]
    outputs = [o.get('type', '?') for o in bot.get('outputs', [])]

    out_types     = {o.get('type') for o in bot.get('outputs', [])}
    weights_list = '\n'.join(f'  - {n}' for n in weight_names)
    needs_ollama = t_mode in ('smart_prompt',) and m_type == 'text_generator'
    ollama_block = (
        '\n  3b. Ensure Ollama is running (required by this brain type):\n'
        '        ollama serve\n'
        '      Pull the model if you haven\'t yet:\n'
        '        ollama pull llava\n'
    ) if needs_ollama else ''

    printer_block = (
        '\n'
        'Thermal printer setup (if you have a USB thermal printer)\n'
        '---------------------------------------------------------\n'
        '  1. Plug the printer into a USB port on the Pi.\n'
        '  2. Find vendor:product ID:  lsusb\n'
        '  3. Set USB permissions (replace XXXX:YYYY with your IDs):\n'
        '       echo \'SUBSYSTEM=="usb", ATTRS{idVendor}=="XXXX", ATTRS{idProduct}=="YYYY", MODE="0666"\' \\\n'
        '         | sudo tee /etc/udev/rules.d/99-thermal-printer.rules\n'
        '       sudo udevadm control --reload-rules && sudo udevadm trigger\n'
        '  4. Unplug and reconnect the printer USB cable.\n'
        '  5. If your printer is NOT 0485:5741 (default for MC206H),\n'
        '     set environment vars before running:\n'
        '       PRINTER_VID=0xXXXX PRINTER_PID=0xYYYY python3 run.py\n'
    ) if 'printer' in out_types else ''

    speaker_block = (
        '\n'
        'Voice setup (one-time, for natural speech)\n'
        '------------------------------------------\n'
        '  mkdir -p ~/piper-voices\n'
        '  cd ~/piper-voices\n'
        '  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx\n'
        '  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json\n'
        '\n'
        '  If Piper voice file not found, the bot falls back to espeak.\n'
        '  To use a different voice, set PIPER_MODEL before running:\n'
        '    PIPER_MODEL=~/piper-voices/my-voice.onnx python3 run.py\n'
    ) if 'speaker' in out_types else ''

    return '\n'.join([
        f'Brainy Bot Package — {name}',
        '=' * 60,
        '',
        'This package contains everything needed to run your bot on a',
        'Raspberry Pi (or any Linux/Mac machine with Python 3.11+).',
        '',
        'Contents:',
        '  - run.py            main bot script',
        '  - bot.json          bot configuration',
        weights_list,
        '  - requirements.txt  Python dependencies',
        '  - README.txt        this file',
        '',
        f'Brain type : {m_type} / {t_mode}',
        f'Inputs     : {", ".join(inputs) or "none"}',
        f'Outputs    : {", ".join(outputs) or "none"}',
        '',
        'Setup (one-time)',
        '----------------',
        '  1. Copy this folder to your Pi (scp, USB, or git).',
        '  2. cd into the folder.',
        '  3. Install dependencies:',
        '       pip install -r requirements.txt --break-system-packages',
        ollama_block,
        printer_block,
        speaker_block,
        'Run the bot',
        '-----------',
        '  python3 run.py',
        '',
        'Stop the bot',
        '------------',
        '  Ctrl+C',
        '',
        'Troubleshooting',
        '---------------',
        '  "No audio device": run  arecord -l  to see card numbers.',
        '  "Connection refused on 11434": run  ollama serve  first.',
        '  "Module not found": re-run the pip install step above.',
        '',
    ])


def _pi_zip_bytes(bot: dict) -> bytes:
    """Build the full Pi export ZIP.  Raises ValueError if weights are missing."""
    weights      = _pi_collect_weights(bot)          # [(zip_path, local_Path)]
    weight_names = [z for z, _ in weights]

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('bot.json', json.dumps(bot, indent=2))
        for zip_path, local_path in weights:
            zf.write(local_path, zip_path)
        zf.writestr('run.py',          _PI_RUN_PY)
        zf.writestr('requirements.txt', _pi_requirements(bot))
        zf.writestr('README.txt',       _pi_readme(bot, weight_names))
    mem.seek(0)
    return mem.getvalue()


def _safe_zip_name(name: str) -> str:
    base = re.sub(r'[^a-zA-Z0-9_-]+', '_', (name or 'my_bot').strip()).strip('_')
    return (base or 'my_bot').lower()


def _sh_quote(value: str) -> str:
    return "'" + (value or '').replace("'", "'\"'\"'") + "'"


def _multipart_zip_body(boundary: str, field_name: str, filename: str, zip_bytes: bytes) -> bytes:
    """Build multipart/form-data body with one file part (stdlib, no requests)."""
    b = boundary.encode('ascii')
    CRLF = b'\r\n'
    disp = (
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'
    ).encode('utf-8')
    return (
        b'--' + b + CRLF
        + disp + CRLF
        + b'Content-Type: application/zip' + CRLF
        + CRLF
        + zip_bytes + CRLF
        + b'--' + b + b'--' + CRLF
    )


@app.post('/api/deploy/pi/download')
def deploy_pi_download():
    data = request.get_json(force=True) or {}
    bot = {
        'id': data.get('id'),
        'type': data.get('type', 'bot'),
        'name': (data.get('name') or 'My Bot').strip(),
        'emoji': data.get('emoji', '🤖'),
        'inputs': data.get('inputs', []),
        'brain': data.get('brain'),
        'outputs': data.get('outputs', []),
        'rules': data.get('rules', []),
    }
    try:
        archive = _pi_zip_bytes(bot)
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400
    filename = f"{_safe_zip_name(bot['name'])}_pi.zip"
    return Response(
        archive,
        mimetype='application/zip',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@app.post('/api/deploy/pi/command')
def deploy_pi_command():
    data = request.get_json(force=True) or {}
    host = (data.get('host') or '').strip()
    user = (data.get('user') or '').strip()
    remote_path = (data.get('remote_path') or '').strip()
    key_path = (data.get('key_path') or '').strip()
    port = int(data.get('port') or 22)
    if not host or not user or not remote_path:
        return jsonify({'ok': False, 'error': 'host, user, and remote_path are required.'})

    bot = {
        'id': data.get('id'),
        'type': data.get('type', 'bot'),
        'name': (data.get('name') or 'My Bot').strip(),
        'emoji': data.get('emoji', '🤖'),
        'inputs': data.get('inputs', []),
        'brain': data.get('brain'),
        'outputs': data.get('outputs', []),
        'rules': data.get('rules', []),
    }
    try:
        archive = _pi_zip_bytes(bot)
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)})

    tmp_dir = Path('data/tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"{_safe_zip_name(bot['name'])}_pi.zip"
    local_zip = (tmp_dir / zip_name).resolve()
    local_zip.write_bytes(archive)

    key_opt = f"-i {_sh_quote(key_path)} " if key_path else ''
    remote_dir = remote_path.rstrip('/') or remote_path
    remote_target = f"{remote_dir}/{zip_name}"
    ssh_cmd = (
        f"ssh -p {port} {key_opt}{user}@{host} "
        f"\"mkdir -p {_sh_quote(remote_dir)}\""
    )
    scp_cmd = (
        f"scp -P {port} {key_opt}{_sh_quote(str(local_zip))} "
        f"{user}@{host}:{_sh_quote(remote_target)}"
    )
    command = f"{ssh_cmd} && {scp_cmd}"

    return jsonify({
        'ok': True,
        'local_zip': str(local_zip),
        'remote_target': f'{user}@{host}:{remote_target}',
        'command': command,
    })


@app.post('/api/deploy/pi/send')
def deploy_pi_send():
    data = request.get_json(force=True) or {}
    host = (data.get('host') or '').strip()
    user = (data.get('user') or '').strip()
    remote_path = (data.get('remote_path') or '').strip()
    key_path = (data.get('key_path') or '').strip()
    port = int(data.get('port') or 22)
    if not host or not user or not remote_path:
        return jsonify({'ok': False, 'error': 'host, user, and remote_path are required.'})

    bot = {
        'id': data.get('id'),
        'type': data.get('type', 'bot'),
        'name': (data.get('name') or 'My Bot').strip(),
        'emoji': data.get('emoji', '🤖'),
        'inputs': data.get('inputs', []),
        'brain': data.get('brain'),
        'outputs': data.get('outputs', []),
        'rules': data.get('rules', []),
    }
    try:
        archive = _pi_zip_bytes(bot)
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)})

    tmp_dir = Path('data/tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"{_safe_zip_name(bot['name'])}_pi.zip"
    local_zip = tmp_dir / zip_name
    local_zip.write_bytes(archive)

    remote_target = f'{user}@{host}:{remote_path.rstrip("/")}/{zip_name}'
    mkdir_cmd = ['ssh', '-p', str(port)]
    scp_cmd = ['scp', '-P', str(port)]
    if key_path:
        mkdir_cmd += ['-i', key_path]
        scp_cmd += ['-i', key_path]
    mkdir_cmd += [f'{user}@{host}', f'mkdir -p "{remote_path}"']
    scp_cmd += [str(local_zip), remote_target]

    try:
        subprocess.run(mkdir_cmd, check=True, capture_output=True, text=True, timeout=15)
        subprocess.run(scp_cmd, check=True, capture_output=True, text=True, timeout=30)
        return jsonify({'ok': True, 'destination': remote_target})
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e)).strip()
        return jsonify({'ok': False, 'error': err[:500] or 'SCP transfer failed.'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})
    finally:
        try:
            local_zip.unlink(missing_ok=True)
        except Exception:
            pass


@app.post('/api/deploy/pi/classroom')
def deploy_pi_classroom():
    data = request.get_json(force=True) or {}
    service_url = (data.get('service_url') or '').strip()
    token = (data.get('token') or '').strip()
    file_field = (data.get('file_field') or 'file').strip() or 'file'
    use_json_receiver = bool(data.get('json_receiver'))
    if not service_url or not token:
        return jsonify({'ok': False, 'error': 'service_url and token are required.'})

    bot = {
        'id': data.get('id'),
        'type': data.get('type', 'bot'),
        'name': (data.get('name') or 'My Bot').strip(),
        'emoji': data.get('emoji', '🤖'),
        'inputs': data.get('inputs', []),
        'brain': data.get('brain'),
        'outputs': data.get('outputs', []),
        'rules': data.get('rules', []),
    }
    try:
        archive = _pi_zip_bytes(bot)
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)})
    zip_name = f"{_safe_zip_name(bot['name'])}_pi.zip"
    deploy_url = service_url.rstrip('/') + '/deploy'

    try:
        if use_json_receiver:
            payload = json.dumps({
                'token': token,
                'filename': zip_name,
                'zip_b64': base64.b64encode(archive).decode('ascii'),
            }).encode('utf-8')
            req = urllib.request.Request(
                deploy_url,
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
        else:
            boundary = '----BrainyBoundary' + uuid.uuid4().hex
            body = _multipart_zip_body(boundary, file_field, zip_name, archive)
            req = urllib.request.Request(
                deploy_url,
                data=body,
                headers={
                    'Content-Type': f'multipart/form-data; boundary={boundary}',
                    'X-Deploy-Token': token,
                },
                method='POST',
            )

        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
            status = getattr(resp, 'status', 200)
        if status != 200:
            return jsonify({'ok': False, 'error': f'Deploy service returned HTTP {status}'})

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get('ok') is False:
                return jsonify({'ok': False, 'error': parsed.get('error', 'Deploy service error.')})
            dest = parsed.get('saved_to', '') if isinstance(parsed, dict) else ''
            return jsonify({'ok': True, 'destination': dest or raw.strip()[:200]})
        except json.JSONDecodeError:
            # Plain-text success (e.g. "Deployment triggered")
            return jsonify({'ok': True, 'destination': raw.strip()[:300]})
    except urllib.error.HTTPError as e:
        err_text = e.read().decode('utf-8', errors='ignore')
        return jsonify({'ok': False, 'error': f'Deploy service rejected request ({e.code}): {err_text[:300]}'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


if __name__ == '__main__':
    print('✓  brainy → http://localhost:5008')
    app.run(debug=False, port=5008, threaded=True, use_reloader=False)
